# Approach — Part B

## Baseline
Use word-level TF-IDF plus cosine similarity.

## Stronger text representation
Use character n-gram TF-IDF. Character features are useful for typos, abbreviations, punctuation changes and concatenated product codes.

## Optional semantic model
Add sentence embeddings to test whether semantic similarity recovers matches with little lexical overlap.

## Pair protocol
Positive pairs come from the same product group. Construct informative negatives, including hard negatives from similar titles.

## Threshold
Sweep the similarity threshold on validation data and report precision, recall, F1, false-positive rate and false-negative rate.

## Error analysis
Save actual false positives and false negatives and explain whether the cause was noisy text, variant ambiguity, missing information or semantic collision.
