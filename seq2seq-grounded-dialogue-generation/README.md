# From Translation to Grounded Dialogue Generation

Three experiments:
1. Basic RNN/LSTM/GRU encoder-decoder for translation.
2. Attention plus decoding-strategy comparison.
3. Dual-source document-grounded Hinglish dialogue generation.

Constraint: do not fine-tune large pretrained transformers or use packaged seq2seq/translation pipelines. Standard framework primitives such as embeddings, recurrent cells, optimizers and training loops are allowed.
