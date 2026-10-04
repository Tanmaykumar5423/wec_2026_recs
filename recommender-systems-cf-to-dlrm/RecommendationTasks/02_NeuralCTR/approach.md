# Approach — Neural CTR

## Data
The dataset contains 13 numerical features, 26 categorical features and a binary click label.

## Preprocessing
Use a train/validation split from the training file. Fit numerical mean and standard deviation on train only. Build one categorical vocabulary per field with an UNK bucket.

## Model progression
1. Logistic-regression sanity baseline.
2. Embedding-based MLP.
3. Deeper MLP with dropout.

Use BCE with logits for stable binary optimization.

## Experiments
Vary one factor at a time: embedding dimension, hidden depth/width, dropout, learning rate and optimizer.

## Evaluation
ROC-AUC and PR-AUC for ranking quality; log loss for probability quality; Accuracy and F1/precision/recall at a chosen threshold. Include calibration if possible.

## Selection
Choose using validation results. Evaluate the original test split only once the design is frozen.
