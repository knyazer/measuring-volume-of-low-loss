# Measuring Volume of Low-Loss Regions

I am testing a hypothesis that the volume of low-loss/high-accuracy regions correspond to the better alignment between model and dataset.

This could be used as a way to evaluate the quality of an inductive biases of the models for a particular dataset.

We start by looking at the classification problems, mainly measuring the accuracy (since that is the ultimate metric of a good model), not the loss.

However, looking at the loss is also an option.

### Experiments:
#### 1. MNIST pairs accuracy vs SGD accuracy

Look at all the pairs of MNIST classification: e.g. 1 vs 2, 0 vs 1, 4 vs 5 etc etc (45 in total). 

For each pair:

Train GC (Guess-And-Check) models: sample a 1_000_000 models, evaluate their accuracy on the training set with a 1000 digit A and 1000 digit B.

Train a bunch (~100) models with SGD on the same training set.

Compare the testing accuracies (on the same holdout set) for both of the models.

Make the plots of the distribution of testing accuracies for both of the models.


