# Approach — Attention + Decoding

## Model
Use Bahdanau additive attention over every encoder state at every decoder step. The decoder state queries the encoder states, softmax produces attention weights, and the weighted sum becomes the current context.

## Experiments
- Same data split as Subtask 1.
- No-attention vs attention.
- Greedy decoding.
- Beam search with beam sizes 3 and 5.
- Optional length normalization for beam scores.

## Analysis
Plot attention matrices for representative examples. Focus on long sentences, repeated words and rare words.

## Interview points
Explain why the fixed final hidden state is a bottleneck, why attention is differentiable retrieval over encoder states, and why beam search can improve likelihood-based generation without guaranteeing factual correctness.
