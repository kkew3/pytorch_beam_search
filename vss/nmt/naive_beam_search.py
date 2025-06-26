from typing import Callable, Literal, NamedTuple, Any

import torch
from torch import Tensor


class ModelOutputs(NamedTuple):
    logits: Tensor
    """The logits of the next token, of shape (batch_size, vocab_size)."""

    past_key_values: Any
    """The past KV caches."""

    encoder_outputs: Any
    """The encoder outputs of current (input_ids, attention_mask)."""


class NormalizedLogProb(float):
    pass


def normalize_log_prob(
    log_prob: float,
    length: int,
    length_normalization: float,
) -> NormalizedLogProb:
    return NormalizedLogProb(log_prob / length**length_normalization)


def naive_beam_search(
    model: Callable[[Tensor, Tensor, Tensor], ModelOutputs],
    bos_token_id: int,
    eos_token_id: int,
    beam_width: int,
    max_length: int,
    length_normalization: float,
    input_ids: Tensor,
    attention_mask: Tensor,
    device: Literal['cpu', 'cuda'],
) -> tuple[list[Tensor], Tensor]:
    """
    Parameters
    ----------
    model : (LongTensor, LongTensor, LongTensor) -> FloatTensor
        The language model, should be callable as `model(input_ids,
        attention_mask, decoder_input_ids)` and return a batch
        of next token logits. Shapes: input_ids---(batch_size,
        enc_seq_len), attention_mask---(batch_size, enc_seq_len),
        decoder_input_ids---(batch_size, dec_seq_len), return
        value---see docstring of `ModelOutputs` named tuple.
    bos_token_id : int
        The beginning-of-sequence token id.
    eos_token_id : int
        The end-of-sequence token id.
    beam_width : int
        The beam width.
    max_length : int
        The max length to sample.
    length_normalization : float
        The length normalization factor. Set to 0 to disable length
        normalization.
    input_ids : LongTensor
        The encoder input ids of shape (batch_size, enc_seq_len).
    attention_mask : LongTensor
        The encoder attention mask of shape (batch_size, enc_seq_len).
    device : 'cpu' | 'cuda'
        The device.

    Returns
    -------
    samples : list[LongTensor]
        The most likely sequences of length batch_size. The i-th element is of
        shape (seq_len_i,) where seq_len_i <= max_length.
    normalized_log_prob : FloatTensor
        The normalized log prob of `sample`, of shape (batch_size,).
    """
    assert beam_width >= 1
    assert max_length >= 1

    # Process the batch of inputs one at a time.
    batch_size = input_ids.size(0)
    final_samples: list[Tensor] = []
    final_normalized_log_probs: list[NormalizedLogProb] = []
    for i in range(batch_size):
        # Form singleton batch of inputs.
        # shape: (1, enc_seq_len)
        curr_input_ids = input_ids[i : i + 1]
        # shape: (1, enc_seq_len)
        curr_attention_mask = attention_mask[i : i + 1]
        curr_beam_width = beam_width
        # curr_beam: a list of candidate (dec_input_ids, unnormalized
        # log_prob).
        curr_beam: list[tuple[list[int], float]] = [([bos_token_id], 0.0)]
        completed_sequences: list[tuple[list[int], NormalizedLogProb]] = []
        for step in range(1, 1 + max_length):
            all_candidates: list[tuple[list[int], float]] = []
            curr_beam_size = len(curr_beam)
            for _ in range(curr_beam_size):
                dec_input_ids, unnormalized_log_prob = curr_beam.pop()
                # The log prob of the next token.
                # shape: (vocab_size,)
                next_log_probs = (
                    model(
                        curr_input_ids,
                        curr_attention_mask,
                        torch.tensor([dec_input_ids], device=device),
                    )
                    .logits.squeeze(0)
                    .log_softmax(0)
                )
                vocab_size = next_log_probs.size(0)
                for j in range(vocab_size):
                    all_candidates.append(
                        (
                            dec_input_ids + [j],
                            unnormalized_log_prob
                            + float(next_log_probs[j].item()),
                        )
                    )
            all_candidates.sort(
                key=lambda x: normalize_log_prob(
                    x[1], step, length_normalization
                ),
                reverse=True,
            )
            topk_candidates = all_candidates[:curr_beam_width]
            for dec_input_ids, unnormalized_log_prob in topk_candidates:
                last_token = dec_input_ids[-1]
                if last_token == eos_token_id:
                    normalized_log_prob = normalize_log_prob(
                        unnormalized_log_prob, step, length_normalization
                    )
                    completed_sequences.append(
                        (dec_input_ids, normalized_log_prob)
                    )
                    curr_beam_width -= 1
                else:
                    curr_beam.append((dec_input_ids, unnormalized_log_prob))
            if curr_beam_width == 0:
                break
        for dec_input_ids, unnormalized_log_prob in curr_beam:
            completed_sequences.append(
                (
                    dec_input_ids,
                    normalize_log_prob(
                        unnormalized_log_prob, max_length, length_normalization
                    ),
                )
            )
        dec_input_ids, normalized_log_prob = max(
            completed_sequences, key=lambda x: x[1]
        )
        # Strip off the bos token at position 0.
        final_samples.append(torch.tensor(dec_input_ids[1:], device=device))
        final_normalized_log_probs.append(normalized_log_prob)
    return (
        final_samples,
        torch.tensor(final_normalized_log_probs, device=device),
    )
