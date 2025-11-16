"""Single MNIST pair experiment for estimating low-loss volume."""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow_datasets as tfds
from tqdm import tqdm


# Configuration ----------------------------------------------------------------

DIGIT_A = 5
DIGIT_B = 8
TRAIN_PER_CLASS = 1000
HOLDOUT_PER_CLASS = 1000
N_RANDOM_SAMPLES = 10_000_000
N_SGD_STEPS = 5_000
BATCH_SIZE = 32
LEARNING_RATE = 3e-3
SEED = 7
PLOT_DIR = Path("plots")
RESULTS_CACHE_PATH = Path("results.pkl")
HISTOGRAM_PATH = PLOT_DIR / "histogram.png"
TRAINING_PLOT_PATH = PLOT_DIR / "training_curves.png"
RANDOM_RANKED_PATH = PLOT_DIR / "random_ranked.png"
TOP_HOLDOUT_PLOT_PATH = PLOT_DIR / "top_random_holdout.png"
HOLDOUT_MEAN_ERROR_PLOT_PATH = PLOT_DIR / "holdout_mean_error_rate.png"
PERCENTILE_HOLDOUT_ERROR_PLOT_PATH = PLOT_DIR / "holdout_error_percentile.png"

PLOT_DIR.mkdir(parents=True, exist_ok=True)


# Cache helpers ---------------------------------------------------------------


def load_results_cache(path: Path = RESULTS_CACHE_PATH) -> Tuple[dict, bool]:
    """Load cached results and guard against corrupt/non-dict payloads."""
    if not path.exists():
        return {}, False
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Could not read cache at {path}: {exc}")
        return {}, True
    if not isinstance(data, dict):
        print(f"Ignoring cache {path} because payload is {type(data)}")
        return {}, True
    return data, False


def save_results_cache(
    payload: dict, path: Path = RESULTS_CACHE_PATH, *, allow_overwrite: bool = True
) -> None:
    """Persist cached results if they are well-formed."""
    if not allow_overwrite:
        print(f"Skipping cache write to {path} (overwrite not allowed).")
        return
    if not isinstance(payload, dict):
        print("Refusing to write cache: payload is not a dict")
        return
    try:
        with path.open("wb") as f:
            pickle.dump(payload, f)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Could not write cache to {path}: {exc}")


# Model ------------------------------------------------------------------------


class TinyMLP(eqx.Module):
    """Two-layer perceptron used for MNIST binary classification."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, key: jax.Array):
        key1, key2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(28 * 28, 32, key=key1)
        self.linear2 = eqx.nn.Linear(32, 2, key=key2)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape(-1)
        x = jax.nn.relu(self.linear1(x))
        return self.linear2(x)


@eqx.filter_jit
def compute_accuracy(model: TinyMLP, images: jax.Array, labels: jax.Array) -> jax.Array:
    """Return accuracy for a batch of images."""

    def predict_single(image, label):
        logits = model(image)
        return jnp.argmax(logits), label

    preds, labels = eqx.filter_vmap(predict_single)(images, labels)
    return jnp.mean(preds == labels)


@eqx.filter_jit
def loss_fn(model: TinyMLP, images: jax.Array, labels: jax.Array) -> jax.Array:
    """Cross-entropy loss between model predictions and integer labels."""

    def loss_single(image, label):
        logits = model(image)
        return optax.softmax_cross_entropy_with_integer_labels(logits, label)

    losses = eqx.filter_vmap(loss_single)(images, labels)
    return jnp.mean(losses)


@eqx.filter_jit
def train_step(
    model: TinyMLP,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    images: jax.Array,
    labels: jax.Array,
) -> Tuple[TinyMLP, optax.OptState, jax.Array]:
    """One SGD step with Adam optimizer."""
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, images, labels)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# Data loading -----------------------------------------------------------------


def apply_random_label_noise(
    split, fraction: float, base_seed: int, *, seed_offset: int
):
    """Randomly reassign labels for a fraction of the split."""
    if fraction <= 0:
        return split
    images, labels = split
    labels = np.array(labels, copy=True)
    n = len(labels)
    fraction = min(1.0, max(0.0, fraction))
    count = int(math.ceil(fraction * n))
    if count == 0:
        return split
    rng = np.random.default_rng(base_seed + seed_offset)
    indices = rng.choice(n, size=count, replace=False)
    labels[indices] = rng.integers(0, 2, size=count, dtype=labels.dtype)
    return images, labels


def load_mnist_pair(
    digit_a: int,
    digit_b: int,
    train_per_class: int,
    holdout_per_class: int,
    seed: int,
    random_label_frac: float = 0.0,
):
    """Return balanced train/holdout splits for two MNIST digits."""

    def _collect(dataset, per_class_limit, seed_offset):
        buckets = {digit_a: [], digit_b: []}
        for image, label in tfds.as_numpy(dataset):
            label = int(label)
            if label not in buckets:
                continue
            if len(buckets[label]) >= per_class_limit:
                continue
            buckets[label].append(np.squeeze(image).astype(np.float32) / 255.0)
            if all(len(v) >= per_class_limit for v in buckets.values()):
                break

        rng = np.random.default_rng(seed + seed_offset)
        for digit in buckets:
            arr = np.stack(buckets[digit])
            perm = rng.permutation(len(arr))
            buckets[digit] = arr[perm]
        return buckets

    def _build_split(buckets, per_class, seed_offset):
        rng = np.random.default_rng(seed + seed_offset)
        images = []
        labels = []
        for digit in (digit_a, digit_b):
            arr = buckets[digit][:per_class]
            label = 0 if digit == digit_a else 1
            images.append(arr)
            labels.append(np.full(len(arr), label, dtype=np.int32))
        images = np.concatenate(images)
        labels = np.concatenate(labels)
        perm = rng.permutation(len(images))
        return images[perm], labels[perm]

    train_ds = tfds.load("mnist", split="train", as_supervised=True)
    test_ds = tfds.load("mnist", split="test", as_supervised=True)

    train_buckets = _collect(train_ds, train_per_class, seed_offset=0)
    train_split = _build_split(train_buckets, train_per_class, seed_offset=1)

    holdout_buckets = _collect(test_ds, holdout_per_class, seed_offset=2)
    holdout_split = _build_split(holdout_buckets, holdout_per_class, seed_offset=3)
    train_split = apply_random_label_noise(
        train_split, random_label_frac, seed, seed_offset=4
    )
    holdout_split = apply_random_label_noise(
        holdout_split, random_label_frac, seed, seed_offset=5
    )
    return train_split, holdout_split


# Experiment helpers -----------------------------------------------------------


def count_parameters(model: TinyMLP) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return sum(leaf.size for leaf in leaves)


@eqx.filter_jit
def _batch_model_accuracies(
    model_keys: jax.Array,
    images: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    """Return accuracies for each initialization key in `model_keys`."""

    def accuracy_for_key(model_key):
        model = TinyMLP(model_key)
        return compute_accuracy(model, images, labels)

    return jax.vmap(accuracy_for_key)(model_keys)


def sample_batch_of_accs(
    key: jax.Array,
    images: jax.Array,
    labels: jax.Array,
    *,
    batch_size: int = 10_000,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Sample a batch of random models and return their accuracies."""
    next_key, batch_seed = jax.random.split(key)
    batch_keys = jax.random.split(batch_seed, batch_size)
    batch_accs = _batch_model_accuracies(batch_keys, images, labels)
    return batch_accs, batch_keys, next_key


def sample_random_accuracies(
    n_samples: int,
    images: np.ndarray,
    labels: np.ndarray,
    key: jax.Array,
    *,
    batch_size: int = 1_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample random models and record their accuracies on the given split."""
    if n_samples <= 0:
        return np.array([]), np.empty((0, 2), dtype=np.uint32)
    images_jax = jnp.array(images)
    labels_jax = jnp.array(labels)
    rng_key = key
    accs = []
    stored_keys = []
    steps = n_samples // batch_size
    for i in tqdm(range(steps)):
        batch_accs, batch_keys, rng_key = sample_batch_of_accs(
            rng_key, images_jax, labels_jax, batch_size=batch_size
        )
        accs.append(np.array(batch_accs))
        stored_keys.append(np.array(batch_keys))
    return np.concatenate(accs), np.concatenate(stored_keys)


def evaluate_keys_on_data(
    model_keys: np.ndarray, dataset, *, batch_size: int = 512
) -> np.ndarray:
    """Evaluate a list of random seeds on the provided dataset (batched)."""
    if len(model_keys) == 0:
        return np.array([])
    images, labels = dataset
    images_jax = jnp.array(images)
    labels_jax = jnp.array(labels)
    accs = []
    for start in range(0, len(model_keys), batch_size):
        batch_keys = jnp.array(model_keys[start : start + batch_size], dtype=jnp.uint32)
        batch_accs = _batch_model_accuracies(batch_keys, images_jax, labels_jax)
        accs.append(np.array(batch_accs))
    return np.concatenate(accs)


def train_network(
    model: TinyMLP,
    train_data,
    *,
    n_steps: int,
    batch_size: int,
    lr: float,
    key: jax.Array,
):
    """Train the model with SGD and record metrics every step."""
    train_images, train_labels = train_data
    train_images_jax = jnp.array(train_images)
    train_labels_jax = jnp.array(train_labels)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    history = {
        "steps": [],
        "train_acc": [],
        "train_loss": [],
    }
    n_samples = len(train_images)
    rng = key

    for step in range(1, n_steps + 1):
        rng, subkey = jax.random.split(rng)
        indices = jax.random.randint(subkey, (batch_size,), 0, n_samples)
        batch_images = train_images_jax[indices]
        batch_labels = train_labels_jax[indices]
        model, opt_state, _ = train_step(
            model, opt_state, optimizer, batch_images, batch_labels
        )

        train_acc = compute_accuracy(model, train_images_jax, train_labels_jax)
        train_loss = loss_fn(model, train_images_jax, train_labels_jax)
        history["steps"].append(step)
        history["train_acc"].append(float(train_acc))
        history["train_loss"].append(float(train_loss))

    return model, history


def plot_random_vs_trained(
    random_accs: np.ndarray,
    trained_acc: float,
    save_path: str,
) -> None:
    """High-resolution histogram highlighting the trained model accuracy."""
    plt.figure(figsize=(10, 5))
    plt.hist(
        random_accs,
        bins=100,
        color="steelblue",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.axvline(
        np.mean(random_accs),
        color="black",
        linestyle="--",
        label=f"Random mean {np.mean(random_accs):.3f}",
    )
    plt.axvline(
        trained_acc,
        color="purple",
        linewidth=2.5,
        label=f"Trained {trained_acc:.3f}",
    )
    plt.xlabel("Accuracy on train split")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.title(f"Random initializations vs trained model ({len(random_accs)} samples)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"Saved histogram to {save_path}")


def plot_training_curves(history, save_path: str) -> None:
    """Plot train/test accuracy and loss over optimization steps."""
    steps = np.array(history["steps"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(steps, history["train_acc"], label="Train")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Train accuracy vs steps")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, history["train_loss"], label="Train")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Train loss vs steps")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_random_ranked(
    random_accs: np.ndarray, trained_acc: float, save_path: str
) -> None:
    """Plot sorted random accuracies to visualize top tail."""
    sorted_accs = np.sort(random_accs)[::-1]
    x = np.arange(1, len(sorted_accs) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(x, sorted_accs, color="steelblue")
    plt.axhline(
        trained_acc,
        color="purple",
        linestyle="--",
        label=f"Trained {trained_acc:.3f}",
    )
    plt.xlabel("Random model rank (best on left)")
    plt.ylabel("Accuracy")
    plt.title("Top random initializations vs trained model")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved ranked plot to {save_path}")


def plot_top_holdout_hist(top_sets, trained_acc: float, save_path: str) -> None:
    """Plot holdout accuracy histograms for top random models."""
    n_sets = len(top_sets)
    fig, axes = plt.subplots(
        n_sets, 1, figsize=(10, 4 * n_sets), sharex=True, squeeze=False
    )
    for idx, entry in enumerate(top_sets):
        ax = axes[idx, 0]
        ax.hist(
            entry["accs"],
            bins=60,
            color="teal",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_yscale("log")
        ax.axvline(
            trained_acc,
            color="purple",
            linestyle="--",
            linewidth=2,
            label=f"Trained {trained_acc:.3f}",
        )
        ax.set_ylabel("Count")
        ax.set_title(
            f"Holdout accuracy for top {entry['label']} random models\n"
            f"mean={np.mean(entry['accs']):.4f}, max={np.max(entry['accs']):.4f}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1, 0].set_xlabel("Holdout accuracy")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"Saved top holdout histogram to {save_path}")


def fit_power_tail(
    fractions: np.ndarray,
    mean_error_rates: np.ndarray,
    *,
    tail_max_fraction: float = 0.05,
):
    """
    Fit log(y - baseline) = intercept + slope * log(x) on the top-fraction tail.
    The baseline captures the limiting Bayes factor / irreducible error.
    """
    mask = fractions <= tail_max_fraction
    if np.count_nonzero(mask) < 2:
        return None
    x = fractions[mask]
    y = mean_error_rates[mask]
    positive = y > 0
    if not np.any(positive):
        return None
    # Search a small grid of baselines to find the line that is straightest in log-log.
    y_min = float(np.min(y))
    grid = np.linspace(0.0, y_min, 100)
    best = None
    for baseline in grid:
        y_adj = y - baseline
        if np.any(y_adj <= 0):
            continue
        lnx = np.log(x)
        lny = np.log(y_adj)
        slope, intercept = np.polyfit(lnx, lny, 1)
        residuals = lny - (slope * lnx + intercept)
        mse = float(np.mean(residuals**2))
        if best is None or mse < best["mse"]:
            best = {
                "baseline": baseline,
                "slope": slope,
                "intercept": intercept,
                "mse": mse,
            }
    if best is None:
        return None
    # Extrapolate slightly beyond the observed minimum percentile.
    target_fraction = max(float(np.min(x)) * 0.1, 1e-8)
    predicted_error = best["baseline"] + math.exp(
        best["intercept"] + best["slope"] * math.log(target_fraction)
    )
    return {
        "baseline": best["baseline"],
        "slope": best["slope"],
        "intercept": best["intercept"],
        "mse": best["mse"],
        "tail_mask": mask,
        "target_fraction": target_fraction,
        "predicted_error": predicted_error,
    }


def plot_holdout_mean_error_rate(
    top_sets,
    save_path: str,
    *,
    trained_holdout_acc: float | None = None,
    tail_fit: dict | None = None,
    tail_max_pct: float = 0.05,
) -> dict | None:
    """Plot holdout mean error rate (1 - accuracy) vs top-percent threshold."""
    fractions = np.array([entry["pct"] for entry in top_sets])
    mean_error_rates = 1.0 - np.array([np.mean(entry["accs"]) for entry in top_sets])

    # Ensure monotonic x-axis for plotting.
    order = np.argsort(fractions)
    fractions = fractions[order]
    mean_error_rates = mean_error_rates[order]
    if tail_fit is None:
        tail_fit = fit_power_tail(
            fractions, mean_error_rates, tail_max_fraction=tail_max_pct
        )

    x_percent = fractions * 100
    plt.figure(figsize=(7, 4))
    plt.loglog(
        x_percent,
        mean_error_rates,
        marker="o",
        color="darkred",
        linewidth=2,
        label="Holdout mean error rate",
    )
    plt.xlabel("Top random models retained (%)")
    plt.ylabel("Mean error rate (1 - accuracy)")
    plt.title("Holdout mean error rate vs top random percentile")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.xticks(
        x_percent,
        [f"{pct:.3g}%" for pct in x_percent],
    )
    if trained_holdout_acc is not None:
        trained_err = 1.0 - trained_holdout_acc
        plt.axhline(
            trained_err,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"Trained holdout error {trained_err:.4f}",
        )
    if tail_fit is not None:
        x_tail = fractions[tail_fit["tail_mask"]]
        x_fit = np.logspace(
            math.log10(max(np.min(x_tail), 1e-8)),
            math.log10(max(np.max(fractions), np.min(x_tail))),
            400,
        )
        y_fit = tail_fit["baseline"] + np.exp(
            tail_fit["intercept"] + tail_fit["slope"] * np.log(x_fit)
        )
        plt.loglog(
            x_fit * 100,
            y_fit,
            linestyle="--",
            color="navy",
            linewidth=2,
            label=f"Power-law tail fit (<= {tail_max_pct * 100:.1f}%)",
        )
        predicted_error = tail_fit["predicted_error"]
        marker_x = tail_fit["target_fraction"]
        plt.scatter(
            [marker_x * 100],
            [predicted_error],
            color="orange",
            marker="x",
            s=80,
            label=(
                f"Extrapolated optimum err {predicted_error:.4f} "
                f"(acc {1 - predicted_error:.4f})"
            ),
        )
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"Saved holdout mean error-rate plot to {save_path}")
    return tail_fit


def plot_holdout_percentile_error(
    top_sets, save_path: str, *, window: int = 100, tail_max_pct: float = 0.05
) -> dict | None:
    """
    Plot the holdout error averaged over a window around each percentile threshold.

    For a threshold p, we look at the models nearest to that cutoff (the last
    `window` samples inside the top-p% set) and plot their mean holdout error.
    Uses log-log axes like the mean-error plot for consistency.
    """
    fractions = []
    percentile_errors = []
    for entry in top_sets:
        train_accs = np.array(entry["train_accs"])
        holdout_accs = np.array(entry["accs"])
        if len(train_accs) == 0 or len(train_accs) != len(holdout_accs):
            continue
        fractions.append(entry["pct"])
        window_size = min(window, len(train_accs))
        window_errors = 1.0 - holdout_accs[-window_size:]
        percentile_errors.append(float(np.mean(window_errors)))

    if len(percentile_errors) == 0:
        print("No percentile data available to plot.")
        return

    # Ensure consistent ordering along the x-axis.
    fractions = np.array(fractions)
    order = np.argsort(fractions)
    fractions = fractions[order]
    percentile_errors = np.array(percentile_errors)[order]

    tail_fit = fit_power_tail(
        fractions, percentile_errors, tail_max_fraction=tail_max_pct
    )

    x_percent = fractions * 100
    plt.figure(figsize=(7, 4))
    plt.loglog(
        x_percent,
        percentile_errors,
        marker="s",
        color="darkgreen",
        linewidth=2,
        label="Holdout error at percentile sample",
    )
    if tail_fit is not None:
        x_tail = fractions[tail_fit["tail_mask"]]
        x_fit = np.logspace(
            math.log10(max(np.min(x_tail), 1e-8)),
            math.log10(max(np.max(fractions), np.min(x_tail))),
            400,
        )
        y_fit = tail_fit["baseline"] + np.exp(
            tail_fit["intercept"] + tail_fit["slope"] * np.log(x_fit)
        )
        plt.loglog(
            x_fit * 100,
            y_fit,
            linestyle="--",
            color="navy",
            linewidth=2,
            label=f"Power-law tail fit (<= {tail_max_pct * 100:.1f}%)",
        )
    plt.xlabel("Percentile of random models retained (%)")
    plt.ylabel("Error rate (1 - accuracy)")
    plt.title("Holdout error vs percentile sample (log-log)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.xticks(x_percent, [f"{pct:.3g}%" for pct in x_percent])
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"Saved percentile holdout-error plot to {save_path}")
    return tail_fit


# Main experiment --------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Estimate low-loss volume for MNIST 0 vs 1."
    )
    parser.add_argument(
        "-r",
        "--random-labels",
        type=float,
        default=0.0,
        help="Fraction (0-1) of train/holdout labels to randomize. Example: 0.1",
    )
    args = parser.parse_args()
    random_label_frac = min(1.0, max(0.0, float(args.random_labels)))
    if random_label_frac > 0:
        print(
            f"Applying random labels to {random_label_frac * 100:.2f}% of train+holdout."
        )

    print(f"Comparing digits {DIGIT_A} vs {DIGIT_B}")
    train_data, holdout_data = load_mnist_pair(
        DIGIT_A,
        DIGIT_B,
        TRAIN_PER_CLASS,
        HOLDOUT_PER_CLASS,
        seed=SEED,
        random_label_frac=random_label_frac,
    )
    print(
        f"Train split: {len(train_data[0])} samples | "
        f"Holdout split: {len(holdout_data[0])} samples"
    )

    key = jax.random.PRNGKey(SEED)
    key, random_key = jax.random.split(key)
    random_accs, random_keys = sample_random_accuracies(
        N_RANDOM_SAMPLES, train_data[0], train_data[1], random_key
    )
    print(
        f"Random models (train split): "
        f"mean={np.mean(random_accs):.3f} best={np.max(random_accs):.3f}"
    )

    key, init_key = jax.random.split(key)
    model = TinyMLP(init_key)
    key, train_key = jax.random.split(key)
    trained_model, history = train_network(
        model,
        train_data,
        n_steps=N_SGD_STEPS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        key=train_key,
    )

    final_train_acc = history["train_acc"][-1]
    better_than_trained = np.mean(random_accs >= final_train_acc) * 100
    param_count = count_parameters(trained_model)
    holdout_images, holdout_labels = holdout_data
    holdout_images_jax = jnp.array(holdout_images)
    holdout_labels_jax = jnp.array(holdout_labels)
    trained_holdout_acc = float(
        compute_accuracy(trained_model, holdout_images_jax, holdout_labels_jax)
    )

    print("\nSummary")
    print("-" * 60)
    print(f"Parameters: {param_count}")
    print(f"Random label noise: {random_label_frac * 100:.2f}%")
    print(f"Final train accuracy:  {final_train_acc:.4f}")
    print(f"Final holdout accuracy:{trained_holdout_acc:.4f}")
    print(
        f"Random models better than trained (train split): {better_than_trained:.2f}%"
    )
    plot_random_vs_trained(random_accs, final_train_acc, HISTOGRAM_PATH)
    plot_training_curves(history, TRAINING_PLOT_PATH)
    plot_random_ranked(random_accs, final_train_acc, RANDOM_RANKED_PATH)

    cache, cache_locked = load_results_cache()
    eval_meta = {
        "digit_a": DIGIT_A,
        "digit_b": DIGIT_B,
        "train_per_class": TRAIN_PER_CLASS,
        "holdout_per_class": HOLDOUT_PER_CLASS,
        "seed": SEED,
        "n_random_samples": len(random_accs),
        "random_label_frac": random_label_frac,
    }
    cached_top_sets = cache.get("top_eval_cache", {})
    cache_updated = False
    if cache.get("top_eval_meta") != eval_meta:
        cached_top_sets = {}

    # 20 log-spaced percent thresholds from 0.001% to 10%.
    percent_specs = [
        (f"{pct * 100:.4g}%", float(pct))
        for pct in np.logspace(np.log10(0.00001), np.log10(0.10), num=20)
    ]
    sorted_indices = np.argsort(random_accs)[::-1]
    top_sets = []
    for label, fraction in percent_specs:
        count = max(1, int(math.ceil(fraction * len(random_accs))))
        indices = sorted_indices[:count]
        subset_keys = random_keys[indices]
        cached_entry = cached_top_sets.get(fraction)
        if (
            cached_entry
            and len(cached_entry.get("accs", [])) == count
            and len(cached_entry.get("train_accs", [])) == count
        ):
            holdout_accs = np.array(cached_entry["accs"])
            train_accs = np.array(cached_entry["train_accs"])
            print(f"Using cached evaluation for top {label}")
        else:
            holdout_accs = evaluate_keys_on_data(
                subset_keys, holdout_data, batch_size=256
            )
            train_accs = evaluate_keys_on_data(subset_keys, train_data, batch_size=256)
            cached_top_sets[fraction] = {
                "label": label,
                "pct": fraction,
                "accs": holdout_accs,
                "train_accs": train_accs,
                "count": count,
            }
            cache_updated = True
        top_sets.append(
            {
                "label": label,
                "pct": fraction,
                "accs": holdout_accs,
                "train_accs": train_accs,
            }
        )
        print(
            f"Top {label}: "
            f"train mean={np.mean(train_accs):.4f}, best={np.max(train_accs):.4f} | "
            f"holdout mean={np.mean(holdout_accs):.4f}, best={np.max(holdout_accs):.4f} "
            f"(count={len(holdout_accs)})"
        )

    if cache_updated:
        cache["top_eval_meta"] = eval_meta
        cache["top_eval_cache"] = cached_top_sets
        if cache_locked:
            print(
                "Skipped cache write because existing cache could not be read safely."
            )
        else:
            save_results_cache(cache, allow_overwrite=True)

    fractions = np.array([entry["pct"] for entry in top_sets])
    mean_error_rates = 1.0 - np.array([np.mean(entry["accs"]) for entry in top_sets])
    tail_fit = fit_power_tail(fractions, mean_error_rates, tail_max_fraction=0.05)
    best_sampled_holdout = max(np.max(entry["accs"]) for entry in top_sets)
    if tail_fit is not None:
        error_factor_per_halving = 2 ** tail_fit["slope"]
        print("Power-law tail fit with limiting error (<=5% top samples):")
        limiting_holdout_acc = 1.0 - tail_fit["baseline"]
        print(f"  Limiting error ~{tail_fit['baseline']:.5f}")
        print(f"  Limiting holdout acc ~{limiting_holdout_acc:.5f}")
        print(
            f"  Limiting vs best sampled holdout acc: "
            f"{limiting_holdout_acc:.5f} vs {best_sampled_holdout:.5f}"
        )
        print(
            "  Residual error scales by "
            f"{error_factor_per_halving:.4f} per 2x tighter percentile "
            f"(slope={tail_fit['slope']:.4f})"
        )
        print(
            f"Trained holdout acc: {trained_holdout_acc:.4f} | "
            f"Best sampled holdout acc: {best_sampled_holdout:.4f}"
        )
    else:
        print("Not enough tail points to fit power-law extrapolation.")

    plot_holdout_mean_error_rate(
        top_sets,
        HOLDOUT_MEAN_ERROR_PLOT_PATH,
        trained_holdout_acc=trained_holdout_acc,
        tail_fit=tail_fit,
        tail_max_pct=0.05,
    )
    percentile_tail_fit = plot_holdout_percentile_error(
        top_sets, PERCENTILE_HOLDOUT_ERROR_PLOT_PATH
    )
    if percentile_tail_fit is not None:
        limiting_err = percentile_tail_fit["baseline"]
        limiting_acc = 1.0 - limiting_err
        best_percentile_acc = 1.0 - percentile_tail_fit["predicted_error"]
        print("Percentile power-law tail fit (window avg):")
        print(
            f"  Limiting error ~{limiting_err:.5f} | limiting acc ~{limiting_acc:.5f}"
        )
        print(
            f"  Extrapolated optimum err {percentile_tail_fit['predicted_error']:.5f} "
            f"| acc {best_percentile_acc:.5f} at ~{percentile_tail_fit['target_fraction'] * 100:.4f}%"
        )
    else:
        print("No percentile-tail fit available.")
    plot_top_holdout_hist(top_sets, trained_holdout_acc, TOP_HOLDOUT_PLOT_PATH)


if __name__ == "__main__":
    main()
