from typing import Callable

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


def naive_beam_search(
    model: Callable[[Tensor], Tensor],
    bos_token_id: int,
    eos_token_id: int,
    beam_width: int,
    max_length: int,
) -> Tensor:
    """
    Parameters
    ----------
    model : (LongTensor) -> FloatTensor
        The language model that takes as input the input ids, of shape
        (batch_size, seq_len) and outputs the logits of a batch of next tokens,
        of shape (batch_size, vocab_size).
    bos_token_id : int
        The beginning-of-sequence token id.
    eos_token_id : int
        The end-of-sequence token id.
    beam_width : int
        The beam width.
    max_length : int
        The max length to sample.

    Returns
    -------
    samples : LongTensor
        The most likely sequence, of shape (seq_len,) where seq_len is no
        larger than max_length.
    """
    assert beam_width >= 1
    assert max_length >= 1

    topk_candidates = torch.tensor([bos_token_id]).unsqueeze_(0)
    topk_log_probs = torch.zeros(1)
    # Element i shape: (k_i, seq_len_i).
    completed_candidates = []
    # Element i shape: (k_i,).
    completed_candidate_scores = []
    for _step in range(max_length):
        # topk_candidates shape: (k, _step + 1), where k <= beam_width.
        # log_probs shape: (vocab_size, k).
        log_probs = model(topk_candidates).log_softmax(-1).t()
        # topk_log_probs shape: (k,).
        # cum_log_probs shape: (vocab_size, k).
        cum_log_probs = topk_log_probs + log_probs
        # topk_log_probs shape: (beam_width,).
        # token_ids & seq_ids shape: (beam_width,).
        topk_log_probs, (token_ids, seq_ids) = topk_2d(
            cum_log_probs, beam_width
        )
        # topk_candidates shape: (beam_width, _step + 2).
        topk_candidates = grow_candidates(topk_candidates, token_ids, seq_ids)
        # is_completed shape: (beam_width,).
        is_completed = token_ids == eos_token_id
        completed_candidates.append(topk_candidates[is_completed, 1:])
        completed_candidate_scores.append(topk_log_probs[is_completed])
        # topk_candidates shape: (k, _step + 2), where k <= beam_width.
        topk_candidates = topk_candidates[~is_completed]
        # topk_log_probs shape: (k,).
        topk_log_probs = topk_log_probs[~is_completed]
        # is_completed.sum() <= beam_width.
        beam_width -= is_completed.sum()
        if beam_width == 0:
            break
    if topk_candidates.size(0) > 0:
        completed_candidates.append(topk_candidates[:, 1:])
        completed_candidate_scores.append(topk_log_probs)
    # Get the index of the top-1 sequence.
    j = torch.cat(completed_candidate_scores, 0).argmax(0)
    return get_tensor_by_flattened_index(completed_candidates, j)
