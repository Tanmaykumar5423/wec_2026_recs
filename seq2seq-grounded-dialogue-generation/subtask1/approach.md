# Approach — Basic Seq2Seq

## Data
Use the English-to-target parallel corpus supplied by the assignment. Keep train/validation/test separated.

## Model
Tokenize with PAD, BOS, EOS and UNK. Use an embedding plus GRU encoder. Feed the final hidden state into a GRU decoder. Train with teacher forcing and cross-entropy while ignoring PAD.

## Inference
Generate one token at a time until EOS or a maximum length.

## Evaluation
Compute BLEU on held-out examples. Break down results by sentence length and inspect rare/unseen token failures.

## Purpose
The fixed context vector is the deliberate bottleneck. Task 2 tests whether token-level attention removes that bottleneck.
