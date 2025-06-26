from typing import Callable, Literal

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import pytest


# Dummy model: simple 2-layer language model for tests.
class DummyLM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        x = self.embed(input_ids)  # (batch, seq_len, hidden)
        x = F.gelu(self.l1(x))  # (batch, seq_len, hidden)
        x = self.l2(x)  # (batch, seq_len, vocab_size)
        return x.mean(1)  # (batch, vocab_size)


def topk_2d(inputs: Tensor, k: int) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    if inputs.dim() != 2:
        raise ValueError(f'inputs.dim() must be 2 but got {inputs.dim()}')
    ncol = inputs.size(1)
    values, indices = inputs.reshape(-1).topk(k)
    return values, (indices // ncol, indices % ncol)


def test_topk_2d():
    with pytest.raises(RuntimeError):
        _ = topk_2d(torch.randn(2, 3), k=8)

    values, (ind_i, ind_j) = topk_2d(torch.tensor([[1, 5, 3, 2, 4]]), k=2)
    assert (values == torch.tensor([5, 4])).all()
    assert (ind_i == torch.tensor([0, 0])).all()
    assert (ind_j == torch.tensor([1, 4])).all()

    values, (ind_i, ind_j) = topk_2d(torch.tensor([[1, 5, 3, 2, 4]]).t(), k=2)
    assert (values == torch.tensor([5, 4])).all()
    assert (ind_i == torch.tensor([1, 4])).all()
    assert (ind_j == torch.tensor([0, 0])).all()

    values, (ind_i, ind_j) = topk_2d(
        torch.tensor([[97, 74, 48, 57], [43, 72, 89, 52]]), k=2
    )
    assert (values == torch.tensor([97, 89])).all()
    assert (ind_i == torch.tensor([0, 1])).all()
    assert (ind_j == torch.tensor([0, 2])).all()


def grow_candidates(
    candidate_seqs: Tensor,
    token_ids: Tensor,
    seq_ids: Tensor,
) -> Tensor:
    """
    Extend candidate sequences `candidate_seqs` by one token from `token_ids`.

    Parameters
    ----------
    candidate_seqs : LongTensor
        The candidate sequences, of shape (n_seq, seq_len).
    token_ids : LongTensor
        The tokens to append, of shape (k,). Each element is in range [0,
        vocab_size).
    seq_ids : LongTensor
        The sequence ids to grow, of shape (k,). Each element is in range [0,
        n_seq).

    Returns
    -------
    grown_candidates : LongTensor
        The result candidates, of shape (k, seq_len + 1).
    """
    return torch.cat((candidate_seqs[seq_ids], token_ids.unsqueeze(1)), 1)


def test_grow_candidates():
    assert (
        grow_candidates(
            candidate_seqs=torch.tensor([[0]]),
            token_ids=torch.tensor([2, 4]),
            seq_ids=torch.tensor([0, 0]),
        )
        == torch.tensor([[0, 2], [0, 4]])
    ).all()
    assert (
        grow_candidates(
            candidate_seqs=torch.tensor([[0, 2], [0, 4]]),
            token_ids=torch.tensor([7, 3]),
            seq_ids=torch.tensor([1, 1]),
        )
        == torch.tensor([[0, 4, 7], [0, 4, 3]])
    ).all()
    assert (
        grow_candidates(
            candidate_seqs=torch.tensor([[0, 4, 7], [0, 4, 3]]),
            token_ids=torch.tensor([1, 2, 3]),
            seq_ids=torch.tensor([1, 0, 1]),
        )
        == torch.tensor([[0, 4, 3, 1], [0, 4, 7, 2], [0, 4, 3, 3]])
    ).all()


def get_tensor_by_flattened_index(tensors: list[Tensor], index: int) -> Tensor:
    """
    Get the `index`-th tensor from `tensors` as if it were concatenated along
    the first axis, which might not be possible because the dimentions along
    other axes are potentially different. See tests for usage.
    """
    ci = torch.tensor([c.size(0) for c in tensors]).cumsum_(0)
    outer_j = torch.searchsorted(ci, index, right=True)
    inner_j = index - ci[outer_j]
    return tensors[outer_j][inner_j]


def test_get_tensor_by_flattened_index():
    tensors = [
        torch.arange(6).view(2, 3),
        torch.arange(4).unsqueeze_(0),
        torch.arange(8).unsqueeze_(0),
        torch.arange(21).view(3, 7),
    ]
    assert (
        get_tensor_by_flattened_index(tensors, 0) == torch.tensor([0, 1, 2])
    ).all()
    assert (
        get_tensor_by_flattened_index(tensors, 1) == torch.tensor([3, 4, 5])
    ).all()
    assert (get_tensor_by_flattened_index(tensors, 2) == torch.arange(4)).all()
    assert (get_tensor_by_flattened_index(tensors, 3) == torch.arange(8)).all()
    assert (get_tensor_by_flattened_index(tensors, 4) == torch.arange(7)).all()
    assert (
        get_tensor_by_flattened_index(tensors, 5) == torch.arange(7, 14)
    ).all()
    assert (
        get_tensor_by_flattened_index(tensors, 6) == torch.arange(14, 21)
    ).all()


class NormalizedLogProb(float):
    pass


def normalize_log_prob(
    log_prob: float,
    length: int,
    length_normalization: float,
) -> NormalizedLogProb:
    return NormalizedLogProb(log_prob / length**length_normalization)


def naive_beam_search(
    model: Callable[[Tensor, Tensor, Tensor], Tensor],
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
    model : (LongTensor, LongTensor, LongTensor, LongTensor) -> FloatTensor
        The language model, should be callable as `model(input_ids,
        attention_mask, decoder_input_ids)` and return a batch
        of next token logits. Shapes: input_ids---(batch_size,
        enc_seq_len), attention_mask---(batch_size, enc_seq_len),
        decoder_input_ids---(batch_size, dec_seq_len), return
        value---(batch_size, vocab_size).
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
    final_samples = []
    final_normalized_log_probs = []
    for i in range(batch_size):
        # Form singleton batch of inputs.
        # shape: (1, enc_seq_len)
        curr_input_ids = input_ids[i : i + 1]
        # shape: (1, enc_seq_len)
        curr_attention_mask = attention_mask[i : i + 1]
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
                    .squeeze_(0)
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
            topk_candidates = all_candidates[:beam_width]
            for dec_input_ids, unnormalized_log_prob in topk_candidates:
                last_token = dec_input_ids[-1]
                if last_token == eos_token_id:
                    normalized_log_prob = normalize_log_prob(
                        unnormalized_log_prob, step, length_normalization
                    )
                    completed_sequences.append(
                        (dec_input_ids, normalized_log_prob)
                    )
                    beam_width -= 1
                else:
                    curr_beam.append((dec_input_ids, unnormalized_log_prob))
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
