# Approach — DLRM

## Architecture
1. Normalize the 13 dense fields using train-only statistics.
2. Map each categorical field to its own embedding table.
3. Send dense inputs through a bottom MLP.
4. Treat the dense output and categorical embeddings as feature vectors.
5. Compute pairwise dot-product interactions.
6. Concatenate the dense vector with the interaction terms.
7. Use a top MLP to produce the click logit.

## Ablation plan
- Standard DLRM.
- DLRM without interactions.
- Embedding dimensions 8, 16 and 32.
- Smaller vs larger bottom/top MLP.

## Fairness
Reuse the exact Task 2 split, seed policy and metrics. Do not retune the test set.

## Key interview question
Does the gain from explicit feature interactions justify additional parameters, compute and memory? The answer should be supported by measurements, not assumptions.
