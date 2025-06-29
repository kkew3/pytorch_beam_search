# PyTorch Beam Search

A fully vectorized beam search decoding for seq2seq [`transformers`](https://huggingface.co/docs/transformers/v4.51.3/en/index) implemented in PyTorch.

Features:

- [x] Batched decoding.
- [x] Fully vectorized.
- [x] KV caching.
- [x] Sequence decoding terminates on EOS.
- [x] With length penalty/normalization.
- [x] Hardware accelerator enabled (e.g. cuda).
