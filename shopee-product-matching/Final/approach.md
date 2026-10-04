# Approach — Final Multimodal Matcher

## Baseline
Let S_text be a text similarity and S_image be an image similarity. Normalize their distributions and compute:
S_final = alpha * S_text + (1 - alpha) * S_image.

Sweep alpha on validation data.

## Learned fusion
For candidate pairs, concatenate text embedding, image embedding, text similarity and image similarity and train a small MLP classifier.

## Candidate generation
Do not compare every pair at large scale. Retrieve candidates with text/image nearest neighbors, then rerank them with the fusion model.

## Hard negatives
Prefer negative pairs that share brand, similar titles or high visual similarity. They force the matcher to learn product-specific distinctions.

## Ablation
Compare text only, image only, equal fusion, tuned fusion and learned fusion.

## Error taxonomy
Variant confusion, noisy title, visual similarity, poor image quality, missing modality and text-image conflict.

## Production discussion
Mention ANN search, embedding caching, batch inference, threshold calibration and monitoring.
