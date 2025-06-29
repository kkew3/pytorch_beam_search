# PyTorch Beam Search

A fully vectorized beam search for seq2seq [`transformers`](https://huggingface.co/docs/transformers/v4.51.3/en/index) implemented in PyTorch.

Features:

- [x] Line-by-line commented with descriptive docstring.
- [x] Batched decoding.
- [x] Fully vectorized.
- [x] KV caching.
- [x] Sequence decoding terminates on EOS.
- [x] With length penalty/normalization.
- [x] Hardware accelerator enabled (e.g. cuda).

## Why this project

In my research I need a fast enough beam search implementation to compute [BLEU score](https://en.wikipedia.org/wiki/BLEU).
However, I didn't find one on the web.

## How I develop this project

Batched and vectorized beam search decoding is tricky to implement.
Therefore, I start from a [naive implementation](./beam_search/naive_beam_search.py) which is neither batched nor vectorized, but simple enough to ensure correctness.
Then, I prompt GPT-4.1 to give me a [batched implementation](./beam_search/batched_beam_search_gpt.py) until it behaves identical to my naive implementation on real data and pretrained models.
Finally, I manually refactor GPT's implementation and add on more efficiency like KV caching, and produce the [final version](./beam_search/batched_beam_search.py), while maintaining invariance of the decoding results.
I profile the code to ensure changes that bring more complexity but only marginal speedup are not merged into the codebase.

## How to use this project

Since basically every application of beam search is slightly different from others, there might be no such one-for-all implementation that can be used directly.
Hence, read the code, either the [naive implementation](./beam_search/naive_beam_search.py) or the [final version](./beam_search/batched_beam_search.py), and adapt it to your need.

## References

DeepLearning.AI courses:

- [C5W3L03 Beam Search](https://www.youtube.com/watch?v=RLWuzLLSIgw).
- [C5W3L04 Refining Beam Search](https://www.youtube.com/watch?v=gb__z7LlN_4).

Papers:

- [Freitag & Al-Onaizan (2017)](https://arxiv.org/pdf/1702.01806): "Beam Search Strategies for Neural Machine Translation".

## Similar projects

- [budzianowski/PyTorch-Beam-Search-Decoding](https://github.com/budzianowski/PyTorch-Beam-Search-Decoding)
- [jojonki/BeamSearch](https://github.com/jojonki/BeamSearch)
- [jarobyte91/pytorch_beam_search](https://github.com/jarobyte91/pytorch_beam_search)

## License

MIT.
