# Approach — Part C

## Baseline
Use a pretrained CNN such as ResNet and take penultimate-layer activations as image embeddings.

## Stronger experiment
Compare with a pretrained ViT or CLIP-style image representation.

## Similarity
L2-normalize embeddings and compare using cosine similarity.

## Retrieval
Build a nearest-neighbor index and visualize the top-K candidates for selected queries.

## Analysis
Inspect:
- same product photographed differently
- different variants/colors
- similar packaging
- similar-looking but different products
- background/crop/quality changes

## Computational awareness
Measure embedding extraction time, embedding size, storage cost and nearest-neighbor search cost. Discuss ANN indexing for scale.
