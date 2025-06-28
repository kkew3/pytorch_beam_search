from typing import Literal, NamedTuple, Any, Protocol

import torch
from torch import Tensor


class ModelOutputs(NamedTuple):
    """
    The model outputs given (input_ids, attention_mask, decoder_input_ids),
    where (input_ids, attention_mask) are the encoder inputs. Shapes:
    input_ids---(batch_size, enc_seq_len), attention_mask---(batch_size,
    enc_seq_len), decoder_input_ids---(batch_size, dec_seq_len).
    """

    logits: Tensor
    """The logits, of shape (batch_size, dec_seq_len, vocab_size)."""

    past_key_values: Any
    """The past KV caches."""

    encoder_last_hidden_state: Any
    """One of the encoder outputs."""

    encoder_hidden_states: Any
    """One of the encoder outputs."""

    encoder_attentions: Any
    """One of the encoder outputs."""


class Model(Protocol):
    def __call__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        encoder_outputs: Any | None,
        decoder_input_ids: Tensor,
        decoder_attention_mask: Tensor,
    ) -> ModelOutputs: ...


@torch.inference_mode()  # type: ignore
def beam_search(
    model: Model,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    beam_width: int,
    max_length: int,
    length_normalization: float,
    input_ids: Tensor,
    attention_mask: Tensor,
    device: Literal['cpu', 'cuda'],
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Parameters
    ----------
    model : (LongTensor, LongTensor, LongTensor) -> FloatTensor
        The language model, should be callable as `model(input_ids,
        attention_mask, decoder_input_ids)` and return the
        model outputs. Shapes: input_ids---(batch_size,
        enc_seq_len), attention_mask---(batch_size, enc_seq_len),
        decoder_input_ids---(batch_size, dec_seq_len), return value---see
        docstring of `ModelOutputs` named tuple.
    bos_token_id : int
        The beginning-of-sequence token id.
    eos_token_id : int
        The end-of-sequence token id.
    pad_token_id : int
        The padding token id.
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
    decoded_input_ids : LongTensor
        The decoded input_ids by beam search, of shape (batch_size, seq_len),
        where seq_len <= max_length.
    decoded_attention_mask : LongTensor
        The attention mask of `decoded_input_ids`, of shape (batch_size,
        seq_len), where 1 means not masked (should attend to; not padding
        token).
    normalized_log_probs : FloatTensor
        The normalized log probabilities of the decoded sequences, of shape
        (batch_size,).
    """
    assert max_length >= 1
    assert beam_width >= 1
    assert length_normalization >= 0.0

    batch_size, enc_seq_len = input_ids.size()

    dim0_indexer = torch.arange(batch_size, device=device)

    # Repeat encoder input for each beam
    # flat_input_ids: # (batch_size * beam_width, enc_seq_len)
    flat_input_ids = (
        input_ids.unsqueeze(1)
        .expand(batch_size, beam_width, enc_seq_len)
        .reshape(batch_size * beam_width, enc_seq_len)
    )
    # flat_attention_mask: # (batch_size * beam_width, enc_seq_len)
    flat_attention_mask = (
        attention_mask.unsqueeze(1)
        .expand(batch_size, beam_width, enc_seq_len)
        .reshape(batch_size * beam_width, enc_seq_len)
    )

    # Since `flat_input_ids` and `flat_attention_mask` never change, we may
    # cache the encoder_outputs
    encoder_outputs = None

    # Initial decoded input_ids (batch_size * beam_width, 1)
    # decoder_input_ids: (batch_size * beam_width, cur_len), initially cur_len=1
    decoder_input_ids = torch.full(
        (batch_size * beam_width, 1),
        bos_token_id,
        dtype=torch.long,
        device=device,
    )

    # cumulative unnormalized log probs: (batch_size * beam_width,)
    cum_log_probs = torch.zeros(batch_size * beam_width, device=device)
    # But -- only the first beam per example gets to start, others to -inf so
    # they're inactive.
    # mask: (batch_size * beam_width,)
    mask = torch.arange(beam_width, device=device).repeat(batch_size) != 0
    cum_log_probs[mask] = float('-inf')

    # Flag for finished beams (batch_size * beam_width,)
    is_finished = torch.zeros(
        batch_size * beam_width, dtype=torch.bool, device=device
    )

    # Also, for length normalization: track length up to EOS (exclusive) for
    # each beam.
    lengths = torch.ones(
        batch_size * beam_width, dtype=torch.long, device=device
    )  # Include BOS.

    seq_len = 0
    for cur_len in range(1, 1 + max_length):
        seq_len = cur_len

        # decoder_attention_mask: (batch_size * beam_width, cur_len)
        decoder_attention_mask = (
            torch.arange(cur_len, device=device) < lengths.unsqueeze(1)
        ).long()

        outputs = model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        # Cache the encoder_outputs.
        encoder_outputs = (
            outputs.encoder_last_hidden_state,
            outputs.encoder_hidden_states,
            outputs.encoder_attentions,
        )

        # outputs.logits: (batch_size * beam_width, cur_len, vocab_size)
        # logits: (batch_size * beam_width, vocab_size)
        logits = outputs.logits[:, -1]  # -1: take last token's logits
        # log_probs: # (batch_size * beam_width, vocab_size)
        log_probs = logits.log_softmax(-1)

        # For finished beams, force only one next token: PAD, to keep them
        # alive and not affecting topk result.
        log_probs[is_finished, :] = float('-inf')
        log_probs[is_finished, pad_token_id] = 0.0

        # The vocabulary size.
        vocab_size = logits.size(-1)
        # Unnormalized log probs.
        # next_scores: (batch_size * beam_width, vocab_size)
        next_scores = cum_log_probs.unsqueeze(-1) + log_probs
        # Normalized log probs.
        # next_norm_scores: (batch_size * beam_width, vocab_size)
        next_norm_scores = next_scores / (
            lengths.pow(length_normalization).unsqueeze_(-1)
        )

        # Now, for each batch example, we want top beam_width out of
        # (beam_width * vocab_size) candidates.

        # topk_norm_scores (top-k normalized scores) will be used after the
        # max_length loop.
        #
        # topk_norm_scores: (batch_size, beam_width)
        # topk_indices: (batch_size, beam_width)
        topk_norm_scores, topk_indices = torch.topk(
            next_norm_scores.view(batch_size, beam_width * vocab_size),
            beam_width,
            dim=-1,
        )
        # topk_scores (top-k unnormalized scores): (batch_size, beam_width)
        topk_scores = next_scores.view(batch_size, beam_width * vocab_size)[
            dim0_indexer.unsqueeze(-1), topk_indices
        ]
        # cum_log_probs: (batch_size * beam_width,)
        cum_log_probs.copy_(topk_scores.view(-1))
        # Unravel which beam & which token.
        # beam_indices & token_indices: (batch_size, beam_width)
        beam_indices = topk_indices // vocab_size  # elem < beam_width
        token_indices = topk_indices % vocab_size  # elem < vocab_size

        # gather_beam_idx: indices for gathering from (batch_size * beam_width)
        # sources.
        offset = dim0_indexer * beam_width
        # gather_beam_idx: (batch_size * beam_width,)
        gather_beam_idx = (offset.unsqueeze_(1) + beam_indices).view(-1)

        # decoder_input_ids: (batch_size, beam_width, cur_len)
        decoder_input_ids = decoder_input_ids[gather_beam_idx]
        # is_finished: (batch_size * beam_width,)
        is_finished = is_finished[gather_beam_idx]
        # lengths: (batch_size * beam_width,)
        lengths = lengths[gather_beam_idx]

        # next_tokens: (batch_size * beam_width,)
        next_tokens = token_indices.view(-1)

        # Append the new tokens to decoder_input_ids
        # decoder_input_ids: (batch_size * beam_width, cur_len + 1)
        decoder_input_ids = torch.cat(
            (decoder_input_ids, next_tokens.unsqueeze(1)), 1
        )

        # is_finished: (batch_size, beam_width,)
        is_finished |= next_tokens == eos_token_id

        # If all beams finish for all examples in the batch, terminate early
        if is_finished.all():
            break

        # Don't update length in the last iteration, as that would overcount
        # lengths for sequences that do not end in max_length steps.
        if cur_len < max_length:
            # Only increment length for not finished
            lengths += (~is_finished).long()

    # Strip off BOS
    # decoder_input_ids: (batch_size * beam_width, seq_len) where seq_len <=
    # max_length
    decoder_input_ids = decoder_input_ids[:, 1:]
    # (batch_size, beam_width, seq_len)
    decoder_input_ids = decoder_input_ids.view(batch_size, beam_width, seq_len)
    # (batch_size, beam_width, seq_len)
    decoder_attention_mask = decoder_attention_mask.view(
        batch_size, beam_width, seq_len
    )
    # Best sequence indices in each beam
    # best_norm_scores: (batch_size,)
    best_norm_scores, j = topk_norm_scores.max(1)
    best_decoder_input_ids = decoder_input_ids[dim0_indexer, j]
    best_decoder_attention_mask = decoder_attention_mask[dim0_indexer, j]
    return best_decoder_input_ids, best_decoder_attention_mask, best_norm_scores
