# Measuring Volume of Low-Loss Regions

Experiment to measure the volume of low-loss regions in parameter space by sampling random initializations of a small CNN on MNIST binary classification (0 vs 1).

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python measure_volume.py
```

If `results.pkl` exists from a previous run, the script will prompt you to load the stored results instead of rerunning the full experiment. This is useful for generating new plots or evaluating top models on the holdout set without retraining.

## What it does

1. **Loads MNIST** - Filters for digits 0 and 1 only (~13k samples), splits into:
   - Train set (60%) - For training the model
   - Eval set (20%) - For evaluating random initializations and monitoring training
   - Holdout set (20%) - For final comparison of best random vs trained model
2. **Samples 1M random initializations** - Creates 1 million randomly initialized CNNs and evaluates them on the eval set
3. **Trains a network** - Trains one network for 20 epochs, monitoring on the eval set
4. **Holdout comparison** - Evaluates:
   - Top 100 random models on the holdout set
   - Top 10,000 random models on the holdout set
   - Trained model on the holdout set
5. **Visualizes results** - Creates plots showing:
   - Distribution of random initialization accuracies with theoretical random classifier baseline
   - Training progress compared to random baseline
   - Holdout performance comparison of top random models vs trained model

## Architecture

Tiny CNN with ~922 parameters:
- Conv 1 channels → 4 channels (3x3 kernel): 40 params
- MaxPool 4x4 (stride 4): 0 params
- Linear 144 → 6: 870 params
- Linear 6 → 2 (no bias): 12 params
- ReLU activations

## Output

- `results.pkl` - Pickled results dictionary with accuracies, losses, training history, and top models' holdout performance
- `histogram.png` - Distribution of random initialization accuracies (log scale y-axis) with theoretical random classifier baseline and trained model marker
- `histogram_mistakes_log.png` - Distribution of mistakes in log scale with theoretical baseline and trained model marker
- `histogram_loss_log.png` - Distribution of losses in log scale with theoretical expected loss for uniform random classifier
- `histogram_error_rate_logscale.png` - Zoomed view of error rates 0-5% with log scale x-axis, theoretical baseline, and individual samples
- `top_models_holdout.png` - **NEW:** Overlapping histograms comparing holdout performance of top 100 vs top 10,000 random models, plus trained model marker
- `results.png` - Combined visualization with 6 plots:
  - Random accuracy distribution
  - Training progress (accuracy)
  - Training progress (loss)
  - Volume of better random initializations over training
  - Volume of better random (log scale)
  - Test accuracy vs loss correlation
- Console output with detailed statistics including top models' holdout performance

## PAC Generalization Bound

The script computes a PAC (Probably Approximately Correct) bound on generalization error using parameter counting:

**Bound:** With probability ≥ (1 - δ), the test error is at most:
```
test_error ≤ train_error + √((log|H| + log(1/δ)) / (2m))
```

Where:
- `|H|` is the hypothesis class size (≈ 2^(n_params × 32) for float32 parameters)
- `m` is the number of training samples
- `δ = 0.5` for 50% confidence

**Parameters:** The SimpleCNN has **~922 parameters**:
- Conv layer: 40 params (1→4 channels, 3×3 kernel)
- Linear1: 870 params (144→6)
- Linear2: 12 params (6→2, no bias)

With ~922 parameters and ~7,800 training samples:
- log|H| ≈ 922 × 32 = 29,504 bits
- PAC bound (50% confidence): test_error ≤ train_error + √(29,504 / (2 × 7,800)) ≈ train_error + 1.37

**Note:** Even with this small network, the bound is still vacuous (>1.0), which is typical for naive parameter-counting PAC bounds. The bound doesn't reflect actual generalization - the network will likely achieve <5% test error despite the bound predicting >100% error.

## Expected Results

Random initializations typically achieve ~50% accuracy (random chance for binary classification), with some variation. Training should achieve >95% accuracy even with this tiny 922-parameter network, showing that networks can find low-loss regions despite most of parameter space being at chance level.

The PAC bound will be vacuous (~1.37 + train_error for 50% confidence), but actual test error will be <5%, demonstrating that classical PAC bounds are extremely pessimistic for neural networks.
