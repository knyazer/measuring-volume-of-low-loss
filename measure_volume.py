import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from scipy import stats
import os


class SimpleCNN(eqx.Module):
    """Tiny convolutional network for MNIST binary classification (~922 parameters)."""

    conv1: eqx.nn.Conv2d
    pool: eqx.nn.MaxPool2d
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(1, 4, kernel_size=3, key=key1)
        self.pool = eqx.nn.MaxPool2d(kernel_size=4, stride=4)  # More aggressive pooling
        self.linear1 = eqx.nn.Linear(
            4 * 6 * 6, 6, key=key2
        )  # Much smaller hidden layer
        self.linear2 = eqx.nn.Linear(6, 2, use_bias=False, key=key3)

    def __call__(self, x):
        # x shape: (28, 28) -> add channel dim
        x = x[None, :, :]  # (1, 28, 28)

        # Conv block
        x = self.conv1(x)  # (4, 26, 26)
        x = jax.nn.relu(x)
        x = self.pool(x)  # (4, 6, 6) - 4x4 pooling reduces 26x26 to 6x6

        # Flatten and dense layers
        x = x.reshape(-1)  # (144,)
        x = self.linear1(x)  # (6,)
        x = jax.nn.relu(x)
        x = self.linear2(x)  # (2,)

        return x


def load_mnist_binary(test_split_ratio=0.2):
    """Load MNIST and filter for 0s and 1s only. Split into train, eval (for random sampling), and holdout."""
    ds = tfds.load("mnist", split="train", as_supervised=True)

    # Convert to numpy arrays
    images, labels = [], []
    for image, label in tfds.as_numpy(ds):
        if label in [0, 1]:
            images.append(image.astype(np.float32) / 255.0)
            labels.append(label)

    images = np.array(images).squeeze()  # Remove channel dim
    labels = np.array(labels)

    # Shuffle
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]

    # Split: train (60%), eval (20% - for random sampling), holdout (20% - final comparison)
    n_total = len(images)
    n_holdout = int(n_total * 0.2)
    n_eval = int(n_total * 0.2)
    n_train = n_total - n_holdout - n_eval

    train_images = images[:n_train]
    train_labels = labels[:n_train]
    eval_images = images[n_train : n_train + n_eval]
    eval_labels = labels[n_train : n_train + n_eval]
    holdout_images = images[n_train + n_eval :]
    holdout_labels = labels[n_train + n_eval :]

    print(f"Train set: {len(train_images)} samples")
    print(f"Eval set (for random sampling): {len(eval_images)} samples")
    print(f"Holdout set (final comparison): {len(holdout_images)} samples")
    print(
        f"Class distribution - 0s: {np.sum(train_labels == 0)}, 1s: {np.sum(train_labels == 1)}"
    )

    return (
        (train_images, train_labels),
        (eval_images, eval_labels),
        (holdout_images, holdout_labels),
    )


@eqx.filter_jit
def compute_accuracy(model, images, labels):
    """Compute accuracy on a batch of images."""

    def predict_single(image):
        logits = model(image)
        return jnp.argmax(logits)

    predictions = eqx.filter_vmap(predict_single)(images)
    return jnp.mean(predictions == labels)


@eqx.filter_jit
def loss_fn(model, images, labels):
    """Cross-entropy loss."""

    def loss_single(image, label):
        logits = model(image)
        return optax.softmax_cross_entropy_with_integer_labels(logits, label)

    losses = eqx.filter_vmap(loss_single)(images, labels)
    return jnp.mean(losses)


@eqx.filter_jit
def train_step(model, opt_state, optimizer, images, labels):
    """Single training step."""
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, images, labels)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def sample_random_accuracies(
    n_samples, test_data, key, data_batch_size=None, model_batch_size=150
):
    """Sample random network initializations and compute their accuracies and losses."""
    test_images, test_labels = test_data
    test_images_jax = jnp.array(test_images)
    test_labels_jax = jnp.array(test_labels)

    # Use smaller subset for speed during random sampling
    if data_batch_size is None:
        data_batch_size = 1_000_000_000
    subset_size = min(data_batch_size, len(test_images))
    test_images_subset = test_images_jax[:subset_size]
    test_labels_subset = test_labels_jax[:subset_size]

    @eqx.filter_jit
    def evaluate_batch(keys):
        """Evaluate a batch of models in parallel using vmap."""

        def init_and_evaluate(single_key):
            model = SimpleCNN(single_key)
            acc = compute_accuracy(model, test_images_subset, test_labels_subset)
            loss = loss_fn(model, test_images_subset, test_labels_subset)
            return acc, loss

        return eqx.filter_vmap(init_and_evaluate)(keys)

    accuracies = []
    losses = []
    n_batches = (n_samples + model_batch_size - 1) // model_batch_size

    print(
        f"\nSampling {n_samples} random initializations in batches of {model_batch_size}..."
    )
    for batch_idx in tqdm(range(n_batches)):
        # Generate batch of keys
        batch_size = min(model_batch_size, n_samples - batch_idx * model_batch_size)
        keys = jax.random.split(key, batch_size + 1)
        key = keys[0]
        batch_keys = keys[1:]

        # Evaluate batch
        batch_accs, batch_losses = evaluate_batch(batch_keys)
        accuracies.extend([float(acc) for acc in batch_accs])
        losses.extend([float(loss) for loss in batch_losses])

    accuracies = np.array(accuracies)
    losses = np.array(losses)

    # Return accuracies, losses, and the test set size used
    return accuracies, losses, subset_size


def count_parameters(model):
    """Count the total number of trainable parameters in the model by traversing the pytree."""
    # Get all leaves of the pytree
    leaves = jax.tree_util.tree_leaves(model)

    # Count parameters in array leaves
    total = 0
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray):
            total += leaf.size

    return total


def compute_pac_bound(
    n_params, n_train_samples, train_error, delta=0.5, bits_per_param=32
):
    """
    Compute PAC generalization bound using parameter counting.

    PAC bound: With probability at least (1 - delta), the generalization error is at most:
        gen_error ≤ train_error + sqrt((log|H| + log(1/delta)) / (2m))

    Where:
        - |H| is the size of hypothesis class
        - For neural networks: log|H| ≈ n_params × bits_per_param (assuming each param is quantized)
        - m is the number of training samples
        - delta is the confidence parameter (0.5 for 50% confidence)

    Args:
        n_params: Number of model parameters
        n_train_samples: Number of training samples
        train_error: Training error rate (0-1)
        delta: Confidence parameter (default 0.5 for 50% confidence)
        bits_per_param: Bits per parameter (default 32 for float32)

    Returns:
        Dictionary with bound information
    """
    # Size of hypothesis class (in bits)
    log_H_bits = n_params * bits_per_param

    # log(1/delta) in bits
    log_inv_delta_bits = np.log2(1 / delta)

    # Generalization bound
    sqrt_term = np.sqrt((log_H_bits + log_inv_delta_bits) / (2 * n_train_samples))
    gen_bound = train_error + sqrt_term

    return {
        "n_params": n_params,
        "n_train_samples": n_train_samples,
        "train_error": train_error,
        "delta": delta,
        "confidence": 1 - delta,
        "log_H_bits": log_H_bits,
        "log_inv_delta_bits": log_inv_delta_bits,
        "sqrt_term": sqrt_term,
        "generalization_bound": gen_bound,
    }


def train_network(
    model, train_data, test_data, random_accs, n_epochs=10, batch_size=128, lr=1e-3
):
    """Train the network to find optimum."""
    train_images, train_labels = train_data
    test_images, test_labels = test_data

    train_images_jax = jnp.array(train_images)
    train_labels_jax = jnp.array(train_labels)
    test_images_jax = jnp.array(test_images)
    test_labels_jax = jnp.array(test_labels)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    n_batches = len(train_images) // batch_size

    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    better_than_random = []
    better_than_random_pct = []

    print(f"\nTraining network for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        # Shuffle training data
        indices = np.random.permutation(len(train_images))

        epoch_losses = []
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            batch_indices = indices[start:end]

            batch_images = train_images_jax[batch_indices]
            batch_labels = train_labels_jax[batch_indices]

            model, opt_state, loss = train_step(
                model, opt_state, optimizer, batch_images, batch_labels
            )
            epoch_losses.append(float(loss))

        # Compute accuracies and losses
        train_acc = compute_accuracy(model, train_images_jax, train_labels_jax)
        test_acc = compute_accuracy(model, test_images_jax, test_labels_jax)
        train_loss = loss_fn(model, train_images_jax, train_labels_jax)
        test_loss = loss_fn(model, test_images_jax, test_labels_jax)

        train_accs.append(float(train_acc))
        test_accs.append(float(test_acc))
        train_losses.append(float(train_loss))
        test_losses.append(float(test_loss))

        # Count how many random samples are better than current model
        n_better = np.sum(random_accs >= test_accs[-1])
        pct_better = 100.0 * n_better / len(random_accs)
        better_than_random.append(n_better)
        better_than_random_pct.append(pct_better)

        print(
            f"Epoch {epoch + 1}/{n_epochs} - "
            f"Train Loss: {train_losses[-1]:.4f}, "
            f"Test Loss: {test_losses[-1]:.4f}, "
            f"Train Acc: {train_accs[-1]:.4f}, "
            f"Test Acc: {test_accs[-1]:.4f}, "
            f"Better Random: {n_better:,} ({pct_better:.2f}%)"
        )

    return model, {
        "train_accs": train_accs,
        "test_accs": test_accs,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "better_than_random": better_than_random,
        "better_than_random_pct": better_than_random_pct,
    }


def compute_random_classifier_pmf(n_samples, p=0.5):
    """
    Compute the probability mass function for a random binary classifier.

    For a random classifier making predictions on N samples with m classes,
    the number of correct predictions X ~ Binomial(N, p) where p = 1/m.
    For binary classification, p = 0.5.

    Returns arrays of possible accuracies and their probabilities.
    Uses normal approximation for large N for numerical stability.
    """
    # Use normal approximation for large N (more numerically stable)
    if n_samples > 100:
        # X ~ N(μ, σ²) where μ = Np, σ² = Np(1-p)
        mu = n_samples * p
        sigma = np.sqrt(n_samples * p * (1 - p))

        # Create array of possible number of correct predictions
        # Focus on range within 5 standard deviations
        k_min = max(0, int(mu - 5 * sigma))
        k_max = min(n_samples, int(mu + 5 * sigma))
        k_values = np.arange(k_min, k_max + 1)

        # Use normal approximation with continuity correction
        # P(X = k) ≈ P(k - 0.5 < X < k + 0.5) for continuous normal
        pmf = stats.norm.cdf(k_values + 0.5, mu, sigma) - stats.norm.cdf(
            k_values - 0.5, mu, sigma
        )

        # Convert to accuracies
        accuracies = k_values / n_samples

    else:
        # For small N, use exact binomial
        k_values = np.arange(0, n_samples + 1)
        pmf = stats.binom.pmf(k_values, n_samples, p)
        accuracies = k_values / n_samples

    return accuracies, pmf, k_values


def plot_histogram(
    random_accs, n_test_samples, trained_acc=None, save_path="histogram.png"
):
    """Plot histogram of random accuracies with log scale counts."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    counts, bins, patches = ax.hist(
        random_accs,
        bins=100,
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
        label="Observed",
    )

    # Add theoretical distribution for truly random binary classifier
    theoretical_accs, theoretical_pmf, theoretical_k = compute_random_classifier_pmf(
        n_test_samples
    )

    # For each histogram bin, compute expected count by summing probabilities
    # of all discrete k values that fall within that bin
    theoretical_counts = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        bin_low = bins[i]
        bin_high = bins[i + 1]
        # Find all k values where k/N falls in this bin
        mask = (theoretical_accs > bin_low) & (theoretical_accs <= bin_high)
        # Sum their probabilities and scale by number of samples
        theoretical_counts[i] = np.sum(theoretical_pmf[mask]) * len(random_accs)

    # Plot as step function matching histogram bins
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.step(
        bins[:-1],
        theoretical_counts,
        where="post",
        color="black",
        linewidth=2.5,
        label="Random Classifier (p=0.5)",
        zorder=5,
    )

    ax.axvline(
        np.mean(random_accs),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(random_accs):.4f}",
    )
    ax.axvline(
        np.median(random_accs),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(random_accs):.4f}",
    )
    if trained_acc is not None:
        ax.axvline(
            trained_acc,
            color="purple",
            linestyle="-",
            linewidth=3,
            label=f"Trained: {trained_acc:.4f}",
        )
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_ylabel("Count (log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(
        f"Distribution of Random Initialization Accuracies (n={len(random_accs):,})",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Histogram saved to {save_path}")
    plt.close()


def plot_mistakes_histogram(
    random_accs,
    n_test_samples,
    trained_acc=None,
    save_path="histogram_mistakes_log.png",
):
    """Plot histogram of number of mistakes in log scale."""
    # Convert accuracy to number of mistakes
    n_mistakes = n_test_samples * (1.0 - random_accs)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Create histogram with log scale
    counts, bins, patches = ax.hist(
        n_mistakes,
        bins=100,
        edgecolor="black",
        alpha=0.7,
        color="coral",
        label="Observed",
    )

    # Add theoretical distribution for truly random binary classifier
    # Number of mistakes = N - X where X is number correct
    theoretical_accs, theoretical_pmf, theoretical_k = compute_random_classifier_pmf(
        n_test_samples
    )
    theoretical_mistakes = n_test_samples - theoretical_k

    # For each histogram bin, compute expected count by summing probabilities
    theoretical_counts = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        bin_low = bins[i]
        bin_high = bins[i + 1]
        # Find all k values where n_mistakes falls in this bin
        mask = (theoretical_mistakes > bin_low) & (theoretical_mistakes <= bin_high)
        theoretical_counts[i] = np.sum(theoretical_pmf[mask]) * len(random_accs)

    # Plot as step function
    ax.step(
        bins[:-1],
        theoretical_counts,
        where="post",
        color="black",
        linewidth=2.5,
        label="Random Classifier (p=0.5)",
        zorder=5,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Number of Mistakes", fontsize=12)
    ax.set_ylabel("Count (log scale)", fontsize=12)
    ax.set_title(
        f"Distribution of Mistakes in Random Initializations (n={len(random_accs):,})",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3, which="both")

    # Add statistics
    mean_mistakes = np.mean(n_mistakes)
    median_mistakes = np.median(n_mistakes)
    ax.axvline(
        mean_mistakes,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_mistakes:.1f}",
    )
    ax.axvline(
        median_mistakes,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_mistakes:.1f}",
    )
    if trained_acc is not None:
        trained_mistakes = n_test_samples * (1.0 - trained_acc)
        ax.axvline(
            trained_mistakes,
            color="purple",
            linestyle="-",
            linewidth=3,
            label=f"Trained: {trained_mistakes:.1f}",
        )
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Mistakes histogram (log scale) saved to {save_path}")
    plt.close()


def plot_loss_histogram(
    random_losses, trained_loss=None, save_path="histogram_loss_log.png"
):
    """Plot histogram of losses in log scale."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.hist(
        random_losses,
        bins=100,
        edgecolor="black",
        alpha=0.7,
        color="orangered",
        label="Observed",
    )

    # Add theoretical expected loss for uniform random classifier
    # For binary cross-entropy with uniform probabilities (p=0.5 for each class),
    # expected loss per sample is -log(0.5) = log(2) ≈ 0.693
    uniform_loss = np.log(2)
    ax.axvline(
        uniform_loss,
        color="black",
        linestyle="-",
        linewidth=2.5,
        label=f"Uniform Random (p=0.5): {uniform_loss:.4f}",
        zorder=5,
    )

    ax.axvline(
        np.mean(random_losses),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(random_losses):.4f}",
    )
    ax.axvline(
        np.median(random_losses),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(random_losses):.4f}",
    )
    if trained_loss is not None:
        ax.axvline(
            trained_loss,
            color="purple",
            linestyle="-",
            linewidth=3,
            label=f"Trained: {trained_loss:.4f}",
        )
    ax.set_xlabel("Loss", fontsize=12)
    ax.set_ylabel("Count (log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(
        f"Distribution of Random Initialization Losses (n={len(random_losses):,})",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Loss histogram (log scale) saved to {save_path}")
    plt.close()


def plot_error_rate_logscale(
    random_accs,
    n_test_samples=None,
    trained_acc=None,
    save_path="histogram_error_rate_logscale.png",
):
    """Plot histogram of error rates with log scale x-axis, zoomed to 0-5% error, showing both bins and individual samples."""
    # Convert accuracy to error rate (percentage)
    error_rates = 100.0 * (1.0 - random_accs)

    # Filter to only include error rates between 0% and 5%
    mask = (error_rates >= 0) & (error_rates <= 5.0)
    filtered_errors = error_rates[mask]

    # Sort to find the lowest error rates
    sorted_errors = np.sort(filtered_errors)
    min_error = sorted_errors[0] if len(sorted_errors) > 0 else None

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Use log-spaced bins for better visualization on log scale
    bins = np.logspace(np.log10(max(0.01, sorted_errors[0] * 0.5)), np.log10(5.0), 80)

    # Plot histogram
    hist_counts, _, _ = ax.hist(
        filtered_errors,
        bins=bins,
        edgecolor="black",
        alpha=0.5,
        color="teal",
        label="Histogram",
    )

    # Add theoretical distribution for truly random binary classifier
    if n_test_samples is None:
        # Infer n_test_samples from the granularity of error rates
        # Error rate = (n_mistakes / N) * 100, so granularity is (100 / N)
        unique_errors = np.unique(error_rates[error_rates > 0])
        if len(unique_errors) > 1:
            # Find approximate granularity
            diffs = np.diff(sorted(unique_errors))
            granularity = (
                np.min(diffs[diffs > 0.0001]) if len(diffs[diffs > 0.0001]) > 0 else 0.1
            )
            n_test_samples_inferred = int(np.round(100 / granularity))
        else:
            n_test_samples_inferred = 2500  # default estimate
    else:
        n_test_samples_inferred = n_test_samples

    # Compute theoretical distribution
    theoretical_accs, theoretical_pmf, theoretical_k = compute_random_classifier_pmf(
        n_test_samples_inferred
    )
    theoretical_error_rates = 100.0 * (1.0 - theoretical_accs)

    # For each histogram bin, compute expected count by summing probabilities
    theoretical_bin_counts = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        bin_low = bins[i]
        bin_high = bins[i + 1]
        # Find all k values where error rate falls in this bin
        mask = (theoretical_error_rates > bin_low) & (
            theoretical_error_rates <= bin_high
        )
        theoretical_bin_counts[i] = np.sum(theoretical_pmf[mask]) * len(random_accs)

    # Plot as step function matching histogram bins
    # Filter out zero counts for better visualization on log scale
    nonzero_mask = theoretical_bin_counts > 0
    if np.any(nonzero_mask):
        # Create step plot
        ax.step(
            bins[:-1],
            theoretical_bin_counts,
            where="post",
            color="black",
            linewidth=3,
            label=f"Random Classifier (p=0.5, N={n_test_samples_inferred})",
            zorder=10,
        )

    # Plot individual samples as dots along the bottom
    # Use jitter on y-axis for visibility
    y_jitter = np.random.uniform(
        0,
        np.max(np.histogram(filtered_errors, bins=bins)[0]) * 0.05,
        size=len(filtered_errors),
    )
    ax.scatter(
        filtered_errors,
        y_jitter,
        alpha=0.3,
        s=2,
        color="navy",
        label=f"Samples (n={len(filtered_errors):,})",
    )

    # Highlight the best (lowest error) samples
    if min_error is not None:
        best_threshold = min_error * 1.5  # Show samples within 50% of the minimum
        best_mask = filtered_errors <= best_threshold
        best_errors = filtered_errors[best_mask]
        best_y_jitter = y_jitter[best_mask]
        ax.scatter(
            best_errors,
            best_y_jitter,
            alpha=0.8,
            s=20,
            color="red",
            label=f"Best samples (≤{best_threshold:.3f}%, n={len(best_errors)})",
            zorder=5,
        )

        # Add vertical line at minimum
        ax.axvline(
            min_error,
            color="darkred",
            linestyle="--",
            linewidth=2,
            label=f"Min: {min_error:.4f}%",
            zorder=4,
        )

    if trained_acc is not None:
        trained_error = 100.0 * (1.0 - trained_acc)
        if 0 <= trained_error <= 5.0:
            ax.axvline(
                trained_error,
                color="purple",
                linestyle="-",
                linewidth=3,
                label=f"Trained: {trained_error:.3f}%",
                zorder=3,
            )

    ax.set_xlabel("Error Rate (%)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xscale("log")
    ax.set_xlim(
        max(0.005, sorted_errors[0] * 0.5) if len(sorted_errors) > 0 else 0.01, 5.0
    )
    ax.set_title(
        f"Distribution of Error Rates (0-5%, log scale)\n"
        f"{np.sum(mask):,} of {len(random_accs):,} samples ({100 * np.sum(mask) / len(random_accs):.2f}%)",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Error rate histogram (log scale x-axis) saved to {save_path}")
    if min_error is not None:
        print(
            f"  Minimum error rate: {min_error:.4f}% (accuracy: {100 - min_error:.4f}%)"
        )
    plt.close()


def plot_top_models_holdout(
    top100_accs,
    top10000_accs,
    trained_acc,
    save_path="top_models_holdout.png",
):
    """Plot histogram of top models' performance on holdout set."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Plot histograms with transparency
    ax.hist(
        top10000_accs,
        bins=50,
        edgecolor="black",
        alpha=0.5,
        color="skyblue",
        label=f"Top 10,000 (mean: {np.mean(top10000_accs):.4f})",
    )
    ax.hist(
        top100_accs,
        bins=50,
        edgecolor="black",
        alpha=0.6,
        color="coral",
        label=f"Top 100 (mean: {np.mean(top100_accs):.4f})",
    )

    # Add vertical lines for statistics
    ax.axvline(
        np.mean(top10000_accs),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Top 10k Mean: {np.mean(top10000_accs):.4f}",
    )
    ax.axvline(
        np.mean(top100_accs),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Top 100 Mean: {np.mean(top100_accs):.4f}",
    )
    ax.axvline(
        np.max(top100_accs),
        color="darkred",
        linestyle=":",
        linewidth=2,
        label=f"Best Random: {np.max(top100_accs):.4f}",
    )
    ax.axvline(
        trained_acc,
        color="purple",
        linestyle="-",
        linewidth=3,
        label=f"Trained: {trained_acc:.4f}",
    )

    ax.set_xlabel("Holdout Accuracy", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "Performance of Top Random Models on Holdout Set",
        fontsize=14,
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Top models holdout histogram saved to {save_path}")
    plt.close()


def plot_results(random_accs, training_history, save_path="results.png"):
    """Plot the results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    epochs = range(1, len(training_history["train_accs"]) + 1)

    # Plot 1: Histogram of random accuracies
    ax = axes[0, 0]
    ax.hist(random_accs, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(
        np.mean(random_accs),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(random_accs):.4f}",
    )
    ax.axvline(
        np.median(random_accs),
        color="green",
        linestyle="--",
        label=f"Median: {np.median(random_accs):.4f}",
    )
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Random Accuracies\n(n={len(random_accs):,})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Training accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, training_history["train_accs"], "b-o", label="Train Accuracy")
    ax.plot(epochs, training_history["test_accs"], "r-o", label="Test Accuracy")
    ax.axhline(
        np.max(random_accs),
        color="green",
        linestyle="--",
        label=f"Best Random: {np.max(random_accs):.4f}",
    )
    ax.axhline(
        np.mean(random_accs),
        color="gray",
        linestyle="--",
        label=f"Mean Random: {np.mean(random_accs):.4f}",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training Progress (Accuracy)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Training loss curves
    ax = axes[0, 2]
    ax.plot(epochs, training_history["train_losses"], "b-o", label="Train Loss")
    ax.plot(epochs, training_history["test_losses"], "r-o", label="Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress (Loss)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Number of random samples better than trained model
    ax = axes[1, 0]
    better_counts = training_history["better_than_random"]
    percentages = training_history["better_than_random_pct"]
    ax.plot(epochs, better_counts, "purple", marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("# Random Models Better", color="purple")
    ax.tick_params(axis="y", labelcolor="purple")
    ax.set_title("Volume of Better Random Initializations")
    ax.grid(True, alpha=0.3)

    # Add percentage on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(epochs, percentages, "orange", marker="s", linewidth=2, alpha=0.7)
    ax2.set_ylabel("Percentage (%)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    # Plot 5: Log scale better random models
    ax = axes[1, 1]
    ax.plot(epochs, better_counts, "purple", marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("# Random Models Better (log scale)")
    ax.set_yscale("log")
    ax.set_title("Volume of Better Random (Log Scale)")
    ax.grid(True, alpha=0.3, which="both")

    # Plot 6: Combined accuracy and loss
    ax = axes[1, 2]
    ax_loss = ax.twinx()
    l1 = ax.plot(epochs, training_history["test_accs"], "b-o", label="Test Accuracy")
    l2 = ax_loss.plot(epochs, training_history["test_losses"], "r-s", label="Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy", color="b")
    ax_loss.set_ylabel("Loss", color="r")
    ax.tick_params(axis="y", labelcolor="b")
    ax_loss.tick_params(axis="y", labelcolor="r")
    ax.set_title("Test Accuracy vs Loss")
    ax.grid(True, alpha=0.3)
    # Combine legends
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Results plot saved to {save_path}")
    plt.close()


def main():
    # Configuration
    N_RANDOM_SAMPLES = 100_000
    N_TRAIN_EPOCHS = 20
    SEED = 42

    print("=" * 60)
    print("Measuring Volume of Low-Loss Regions")
    print("=" * 60)

    # Check if results already exist
    results_exist = os.path.exists("results.pkl")
    load_results = False

    if results_exist:
        print("\nFound existing results.pkl file.")
        response = (
            input("Do you want to load stored results instead of rerunning? (y/n): ")
            .strip()
            .lower()
        )
        load_results = response == "y"

    # Set random seeds
    np.random.seed(SEED)
    key = jax.random.PRNGKey(SEED)

    # Load data
    print("\nLoading MNIST data (0s and 1s only)...")
    train_data, eval_data, holdout_data = load_mnist_binary()

    if load_results:
        print("\nLoading stored results...")
        with open("results.pkl", "rb") as f:
            results = pickle.load(f)
        random_accs = results["random_accuracies"]
        random_losses = results["random_losses"]
        training_history = results["training_history"]
        n_eval_samples = len(eval_data[0])

        # We'll still need to evaluate top models on holdout, so continue
        print("Loaded results successfully!")

        # Generate the random sampling key to get consistent model indices
        key, subkey = jax.random.split(key)

        # Need to load the trained model from somewhere or skip training parts
        # For now, we'll just use the stored results
        trained_model = None  # We don't have the model saved
        trained_holdout_acc = results["holdout_comparison"]["trained_acc"]
        trained_holdout_loss = results["holdout_comparison"]["trained_loss"]

        # Check if PAC bound was saved
        if "pac_bound" in results:
            print("\n" + "=" * 60)
            print("PAC Generalization Bound (from saved results):")
            print("=" * 60)
            pac_bound = results["pac_bound"]
            print(f"Model parameters: {pac_bound['n_params']:,}")
            print(f"Training samples: {pac_bound['n_train_samples']:,}")
            print("Bits per parameter: 32 (float32)")
            print(f"log|H| (bits): {pac_bound['log_H_bits']:,}")
            print(
                f"Confidence level: {pac_bound['confidence']:.0%} (δ = {pac_bound['delta']})"
            )
            print(f"\nTraining error: {pac_bound['train_error']:.4f}")
            print(f"Generalization bound term: ±{pac_bound['sqrt_term']:.4f}")
            print(f"PAC bound on test error: ≤ {pac_bound['generalization_bound']:.4f}")
            if pac_bound["generalization_bound"] > 1.0:
                print(f"\nNote: This bound is vacuous (> 1.0), which is typical for")
                print(f"naive parameter-counting PAC bounds on neural networks.")
                print(
                    f"The bound is extremely loose and doesn't reflect actual generalization."
                )
            else:
                test_error = 1.0 - training_history["test_accs"][-1]
                print(f"\nActual test error: {test_error:.4f}")
                print(
                    f"Bound slack: {pac_bound['generalization_bound'] - test_error:.4f}"
                )

        # Check if top models' accuracies are already saved
        if "top100_holdout_accs" in results["holdout_comparison"]:
            print(
                "\nTop models' holdout accuracies found in saved results, loading them..."
            )
            top100_holdout_accs_loaded = np.array(
                results["holdout_comparison"]["top100_holdout_accs"]
            )
            top10000_holdout_accs_loaded = np.array(
                results["holdout_comparison"]["top10000_holdout_accs"]
            )
            skip_top_eval = True
        else:
            print("\nTop models' holdout accuracies not found, will evaluate them...")
            skip_top_eval = False

    else:
        # Sample random initializations (evaluated on eval set)
        key, subkey = jax.random.split(key)
        random_accs, random_losses, n_eval_samples = sample_random_accuracies(
            N_RANDOM_SAMPLES, eval_data, subkey
        )

        # Print statistics
        print("\n" + "=" * 60)
        print("Random Initialization Statistics (on eval set):")
        print("=" * 60)
        print(f"Mean accuracy: {np.mean(random_accs):.4f}")
        print(f"Median accuracy: {np.median(random_accs):.4f}")
        print(f"Std dev: {np.std(random_accs):.4f}")
        print(f"Min accuracy: {np.min(random_accs):.4f}")
        print(f"Max accuracy: {np.max(random_accs):.4f}")
        print(f"\nMean loss: {np.mean(random_losses):.4f}")
        print(f"Median loss: {np.median(random_losses):.4f}")
        print(f"Min loss: {np.min(random_losses):.4f}")
        print(f"Max loss: {np.max(random_losses):.4f}")

        # Count how many are above certain thresholds
        thresholds = [0.6, 0.7, 0.8, 0.9, 0.95]
        print("\nAccuracies above thresholds:")
        for threshold in thresholds:
            count = np.sum(random_accs >= threshold)
            percentage = 100 * count / len(random_accs)
            print(f"  >= {threshold:.2f}: {count:,} ({percentage:.2f}%)")

        # Train a network (evaluated on eval set during training)
        key, subkey_train = jax.random.split(key)
        model = SimpleCNN(subkey_train)
        trained_model, training_history = train_network(
            model, train_data, eval_data, random_accs, n_epochs=N_TRAIN_EPOCHS
        )

        print("\n" + "=" * 60)
        print("Training Results (on eval set):")
        print("=" * 60)
        print(f"Final train accuracy: {training_history['train_accs'][-1]:.4f}")
        print(f"Final eval accuracy: {training_history['test_accs'][-1]:.4f}")
        print(f"Final train loss: {training_history['train_losses'][-1]:.4f}")
        print(f"Final eval loss: {training_history['test_losses'][-1]:.4f}")
        print(
            f"Improvement over best random: "
            f"{training_history['test_accs'][-1] - np.max(random_accs):.4f}"
        )

        # Compute PAC bound
        n_params = count_parameters(trained_model)
        train_error = 1.0 - training_history["train_accs"][-1]
        pac_bound = compute_pac_bound(
            n_params=n_params,
            n_train_samples=len(train_data[0]),
            train_error=train_error,
            delta=0.5,  # 50% confidence
            bits_per_param=32,
        )

        print("\n" + "=" * 60)
        print("PAC Generalization Bound (50% confidence):")
        print("=" * 60)
        print(f"Model parameters: {pac_bound['n_params']:,}")
        print(f"Training samples: {pac_bound['n_train_samples']:,}")
        print(f"Bits per parameter: 32 (float32)")
        print(f"log|H| (bits): {pac_bound['log_H_bits']:,}")
        print(
            f"Confidence level: {pac_bound['confidence']:.0%} (δ = {pac_bound['delta']})"
        )
        print(f"\nTraining error: {pac_bound['train_error']:.4f}")
        print(f"Generalization bound term: ±{pac_bound['sqrt_term']:.4f}")
        print(f"PAC bound on test error: ≤ {pac_bound['generalization_bound']:.4f}")
        if pac_bound["generalization_bound"] > 1.0:
            print(f"\nNote: This bound is vacuous (> 1.0), which is typical for")
            print(f"naive parameter-counting PAC bounds on neural networks.")
            print(
                f"The bound is extremely loose and doesn't reflect actual generalization."
            )
        else:
            print(f"\nActual test error: {1.0 - training_history['test_accs'][-1]:.4f}")
            print(
                f"Bound slack: {pac_bound['generalization_bound'] - (1.0 - training_history['test_accs'][-1]):.4f}"
            )

        # Evaluate trained model on holdout set
        holdout_images, holdout_labels = holdout_data
        holdout_images_jax = jnp.array(holdout_images)
        holdout_labels_jax = jnp.array(holdout_labels)

        trained_holdout_acc = compute_accuracy(
            trained_model, holdout_images_jax, holdout_labels_jax
        )
        trained_holdout_loss = loss_fn(
            trained_model, holdout_images_jax, holdout_labels_jax
        )
        skip_top_eval = False

    # From here on, common code for both paths
    # Find top models by eval set accuracy
    print("\n" + "=" * 60)
    print("Evaluating Top Random Models on Holdout Set")
    print("=" * 60)

    # Get indices of top models
    top_k_values = [100, 10000]
    top_indices = {k: np.argsort(random_accs)[-k:][::-1] for k in top_k_values}

    print(f"\nFound top 100 and top 10,000 random models by eval accuracy")
    print(f"Best eval accuracy: {random_accs[top_indices[100][0]]:.4f}")
    print(f"100th best eval accuracy: {random_accs[top_indices[100][-1]]:.4f}")
    print(f"10,000th best eval accuracy: {random_accs[top_indices[10000][-1]]:.4f}")

    if skip_top_eval:
        # Use loaded results
        print("\nUsing loaded top models' holdout accuracies...")
        top100_holdout_accs = top100_holdout_accs_loaded
        top10000_holdout_accs = top10000_holdout_accs_loaded
    else:
        # Re-initialize top models and evaluate on holdout
        holdout_images, holdout_labels = holdout_data
        holdout_images_jax = jnp.array(holdout_images)
        holdout_labels_jax = jnp.array(holdout_labels)

        # Generate all model keys at once for consistency
        all_keys = jax.random.split(subkey, N_RANDOM_SAMPLES + 1)

        # Evaluate top 100 models
        print(f"\nEvaluating top 100 models on holdout set...")
        top100_holdout_accs = []
        for idx in tqdm(top_indices[100]):
            model_key = all_keys[idx + 1]
            model = SimpleCNN(model_key)
            acc = compute_accuracy(model, holdout_images_jax, holdout_labels_jax)
            top100_holdout_accs.append(float(acc))
        top100_holdout_accs = np.array(top100_holdout_accs)

        # Evaluate top 10,000 models
        print(f"\nEvaluating top 10,000 models on holdout set...")
        top10000_holdout_accs = []
        for idx in tqdm(top_indices[10000]):
            model_key = all_keys[idx + 1]
            model = SimpleCNN(model_key)
            acc = compute_accuracy(model, holdout_images_jax, holdout_labels_jax)
            top10000_holdout_accs.append(float(acc))
        top10000_holdout_accs = np.array(top10000_holdout_accs)

    # Print statistics
    print("\n" + "=" * 60)
    print("HOLDOUT SET COMPARISON:")
    print("=" * 60)
    print(f"\nTop 100 Random Models:")
    print(f"  Mean Accuracy: {np.mean(top100_holdout_accs):.4f}")
    print(f"  Std Accuracy: {np.std(top100_holdout_accs):.4f}")
    print(f"  Best Accuracy: {np.max(top100_holdout_accs):.4f}")
    print(f"  Worst Accuracy: {np.min(top100_holdout_accs):.4f}")
    print(f"\nTop 10,000 Random Models:")
    print(f"  Mean Accuracy: {np.mean(top10000_holdout_accs):.4f}")
    print(f"  Std Accuracy: {np.std(top10000_holdout_accs):.4f}")
    print(f"  Best Accuracy: {np.max(top10000_holdout_accs):.4f}")
    print(f"  Worst Accuracy: {np.min(top10000_holdout_accs):.4f}")
    print(f"\nTrained Model:")
    print(f"  Accuracy: {float(trained_holdout_acc):.4f}")
    print(f"  Loss: {float(trained_holdout_loss):.4f}")
    print(f"\nDifference (Trained - Best Random):")
    print(
        f"  Accuracy: {float(trained_holdout_acc) - np.max(top100_holdout_accs):+.4f}"
    )

    # Plot histograms with trained model markers
    final_test_acc = training_history["test_accs"][-1]
    final_test_loss = training_history["test_losses"][-1]

    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    plot_histogram(
        random_accs, n_eval_samples, final_test_acc, save_path="histogram.png"
    )
    plot_mistakes_histogram(
        random_accs,
        n_eval_samples,
        final_test_acc,
        save_path="histogram_mistakes_log.png",
    )
    plot_loss_histogram(
        random_losses, final_test_loss, save_path="histogram_loss_log.png"
    )
    plot_error_rate_logscale(
        random_accs,
        n_eval_samples,
        final_test_acc,
        save_path="histogram_error_rate_logscale.png",
    )

    # Plot top models' holdout performance
    plot_top_models_holdout(
        top100_holdout_accs,
        top10000_holdout_accs,
        float(trained_holdout_acc),
        save_path="top_models_holdout.png",
    )

    # Save results
    if not load_results or not skip_top_eval:
        # Save if we just ran the experiment or if we evaluated top models
        results = {
            "random_accuracies": random_accs,
            "random_losses": random_losses,
            "training_history": training_history,
            "holdout_comparison": {
                "trained_acc": float(trained_holdout_acc),
                "trained_loss": float(trained_holdout_loss),
                "top100_holdout_accs": top100_holdout_accs.tolist(),
                "top10000_holdout_accs": top10000_holdout_accs.tolist(),
            },
            "config": {
                "n_random_samples": N_RANDOM_SAMPLES,
                "n_train_epochs": N_TRAIN_EPOCHS,
                "seed": SEED,
            },
        }

        # Add PAC bound if we just trained
        if not load_results:
            results["pac_bound"] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in pac_bound.items()
            }

        with open("results.pkl", "wb") as f:
            pickle.dump(results, f)
        print("\nResults saved to results.pkl")

    # Plot training results
    plot_results(random_accs, training_history)

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
