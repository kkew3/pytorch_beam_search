# yapf: disable
from typing import Callable, Literal, NamedTuple, Any

import torch
from torch import Tensor


class ModelOutputs(NamedTuple):
    """
    The model outputs given (input_ids, attention_mask, decoder_input_ids), where (input_ids, attention_mask) are the encoder inputs. Shapes: input_ids---(batch_size, enc_seq_len), attention_mask---(batch_size, enc_seq_len), decoder_input_ids---(batch_size, dec_seq_len).
    """

    logits: Tensor
    """The logits, of shape (batch_size, dec_seq_len, vocab_size)."""

    past_key_values: Any
    """The past KV caches."""

    encoder_outputs: Any
    """The encoder outputs given the encoder inputs."""



def beam_search(
    model: Callable[[Tensor, Tensor, Tensor], ModelOutputs],
    bos_token_id: int,
    eos_token_id: int,
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
        The language model, should be callable as `model(input_ids, attention_mask, decoder_input_ids)` and return the model outputs. Shapes: input_ids---(batch_size, enc_seq_len), attention_mask---(batch_size, enc_seq_len), decoder_input_ids---(batch_size, dec_seq_len), return value---see docstring of `ModelOutputs` named tuple.
    bos_token_id : int
        The beginning-of-sequence token id.
    eos_token_id : int
        The end-of-sequence token id.
    beam_width : int
        The beam width.
    max_length : int
        The max length to sample.
    length_normalization : float
        The length normalization factor. Set to 0 to disable length normalization.
    input_ids : LongTensor
        The encoder input ids of shape (batch_size, enc_seq_len).
    attention_mask : LongTensor
        The encoder attention mask of shape (batch_size, enc_seq_len).
    device : 'cpu' | 'cuda'
        The device.

    Returns
    -------
    decoded_input_ids : LongTensor
        The decoded input_ids by beam search, of shape (batch_size, seq_len), where seq_len <= max_length.
    decoded_attention_mask : LongTensor
        The attention mask of `decoded_input_ids`, of shape (batch_size, seq_len), where 1 means not masked (should attend to; not padding token).
    normalized_log_probs : FloatTensor
        The normalized log probabilities of the decoded sequences, of shape (batch_size,).
    """
    # input_ids: (batch_size, enc_seq_len)
    # attention_mask: (batch_size, enc_seq_len)
    batch_size = input_ids.size(0)
    enc_seq_len = input_ids.size(1)
    vocab_device = device

    # For simplicity, pad to max_length everywhere for decoder
    seq_len = max_length

    # Repeat encoder input for each beam
    flat_input_ids = input_ids.unsqueeze(1).expand(batch_size, beam_width, enc_seq_len).reshape(batch_size * beam_width, enc_seq_len)  # (B*beam, enc_seq_len)
    flat_attention_mask = attention_mask.unsqueeze(1).expand(batch_size, beam_width, enc_seq_len).reshape(batch_size * beam_width, enc_seq_len)  # (B*beam, enc_seq_len)

    # Initial decoded input_ids (B*beam, 1)
    generated = torch.full(
        (batch_size * beam_width, 1),
        bos_token_id,
        dtype=torch.long,
        device=device,
    )  # (B*beam, cur_len), initially cur_len=1

    # cumulative log probs (B*beam,)
    cum_log_probs = torch.zeros(batch_size * beam_width, device=device)  # initialized to 0
    # But -- only the first beam per example gets to start, others to -inf so they're inactive.
    mask = torch.arange(beam_width, device=device).unsqueeze(0).expand(batch_size, beam_width)  # (B,beam)
    init_beam_mask = (mask == 0).reshape(-1)  # (B*beam,)
    cum_log_probs[~init_beam_mask] = float('-inf')

    # Flag for finished beams (B*beam,)
    is_finished = torch.zeros(batch_size * beam_width, dtype=torch.bool, device=device)

    # For each beam, track the generated tokens (list for ragged; but for vectorized we'll always append)
    # Also, for length normalization: track length up to eos for each beam
    lengths = torch.ones(batch_size * beam_width, dtype=torch.long, device=device)  # includes BOS

    # For attention_mask of generated: 1 for non-pad, 0 for padded (batching)
    gen_attention_masks = torch.ones_like(generated, device=device, dtype=torch.long)  # (B*beam, cur_len). Update as generated grows.

    # We'll store the final hypotheses
    # For each example, we track a list of (normalized_log_prob, tokens, attention_mask, length)
    final_hyps = [[] for _ in range(batch_size)]  # list-of-list

    for step in range(1, max_length+1):  # 1 because BOS is already there
        # Model expects: input_ids (B*beam, enc_seq_len), attention_mask (B*beam, enc_seq_len), decoder_input_ids (B*beam, cur_len)
        outputs = model(
            flat_input_ids,         # (B*beam, enc_seq_len)
            flat_attention_mask,    # (B*beam, enc_seq_len)
            generated               # (B*beam, cur_len)
        )
        # outputs.logits: (B*beam, cur_len, vocab_size)
        # logits: (B*beam, vocab_size)
        logits = outputs.logits[:, -1]  # -> take last token's logits
        log_probs = torch.log_softmax(logits, dim=-1)  # (B*beam, vocab_size)

        # For finished beams, force only one next token: EOS, to keep them alive and not perturb
        log_probs[is_finished, :] = float('-inf')
        log_probs[is_finished, eos_token_id] = 0.0

        # For each beam, select beam_width tokens for search (expand all then prune)
        # So for each batch element, there are beam_width beams
        vocab_size = logits.size(-1)
        next_scores = cum_log_probs[:, None] + log_probs  # (B*beam, vocab_size)

        # Now, for each batch example, we want top (beam_width) out of (beam_width*vocab_size) candidates
        #  1. group next_scores as (B, beam, vocab)
        next_scores = next_scores.view(batch_size, beam_width, vocab_size)  # (B, beam, vocab)
        #  2. flatten beam*vocab per batch to (B, beam*vocab)
        next_scores_flat = next_scores.reshape(batch_size, -1)  # (B, beam*vocab)
        #  3. get the top beam_width scores per batch
        topk_scores, topk_indices = torch.topk(next_scores_flat, beam_width, dim=1)  # (B, beam)
        #  4. unravel which beam & which token
        beam_indices = topk_indices // vocab_size   # (B, beam), beam pos in [0,beam_width)
        token_indices = topk_indices % vocab_size   # (B, beam), token pos in [0,vocab_size)

        # Collate new generation state
        # We need to gather beams and append next token
        # Build new generated, cum_log_probs, is_finished, lengths, gen_attention_masks

        # Indices for gathering from (B*beam) source. For batch b, beam_indices[b,:] is in 0..beam_width-1, so:
        gather_beam_idx = (torch.arange(batch_size, device=device) * beam_width).unsqueeze(1) + beam_indices  # (B, beam)
        gather_beam_idx = gather_beam_idx.view(-1)  # (B*beam,)

        # For generated/attn_mask: gather from previous state, append next token/new mask
        generated = generated[gather_beam_idx]                     # (B*beam, cur_len)
        gen_attention_masks = gen_attention_masks[gather_beam_idx] # (B*beam, cur_len)
        cum_log_probs = topk_scores.view(-1)                      # (B*beam,)
        is_finished = is_finished[gather_beam_idx]                # (B*beam,)
        lengths = lengths[gather_beam_idx]                        # (B*beam,)

        next_tokens = token_indices.view(-1, 1)  # (B*beam, 1)

        # Append the new token to generated tensor (grow: (B*beam, step+1))
        generated = torch.cat([generated, next_tokens], dim=1)  # (B*beam, new_len)
        gen_attention_masks = torch.cat([gen_attention_masks, (~is_finished).long().unsqueeze(-1)], dim=1)  # (B*beam, new_len)

        just_finished = (~is_finished) & (next_tokens.squeeze(-1) == eos_token_id)
        is_finished = is_finished | (next_tokens.squeeze(-1) == eos_token_id)

        lengths = lengths + (~is_finished).long()  # Only increment length for not finished (if not finished, it became length+1 by current token)

        # Check if any finished hypotheses need to be stored for each batch item
        for b in range(batch_size):
            # Within batch index b, check all beams (beam_width) for just-finished (by EOS)
            offset = b * beam_width
            for beam_idx in range(beam_width):
                idx = offset + beam_idx
                if just_finished[idx]:
                    # Collect current
                    this_gen = generated[idx]  # (cur_len,)
                    this_mask = gen_attention_masks[idx]  # (cur_len,)
                    this_logprob = cum_log_probs[idx].item()
                    # Length up to EOS (includes BOS; +1 if using the EOS just generated; else lengths[idx])
                    this_len = lengths[idx].item()
                    assert this_len > 0
                    # Length normalization
                    normed_logprob = this_logprob / (this_len ** length_normalization)
                    # Save: (norm_prob, tokens, attn_mask, length)
                    final_hyps[b].append((normed_logprob, this_gen.clone(), this_mask.clone(), this_logprob, this_len))

        # If all beams finished for all batches, terminate early
        if is_finished.all():
            break

    # For beams not yet finished after max_len, treat their current token sequence as finished
    for b in range(batch_size):
        offset = b * beam_width
        for beam_idx in range(beam_width):
            idx = offset + beam_idx
            if not is_finished[idx]:
                # Consider last one as a valid candidate
                this_gen = generated[idx]
                this_mask = gen_attention_masks[idx]
                this_logprob = cum_log_probs[idx].item()
                this_len = lengths[idx].item()
                assert this_len > 0
                normed_logprob = this_logprob / (this_len ** length_normalization)
                final_hyps[b].append((normed_logprob, this_gen.clone(), this_mask.clone(), this_logprob, this_len))

    # Now, pick the best hypothesis for each batch
    out_sequences = []
    out_attention_masks = []
    out_logprobs = []

    for b in range(batch_size):
        # max by normalized logprob
        hyps = final_hyps[b]
        assert len(hyps) > 0
        best = max(hyps, key=lambda z: z[0])
        _, seq, mask, real_logprob, seq_len = best
        # Find where to stop (either at EOS or full length)
        # Drop any pad tokens after EOS if any (sequence may be shorter than max, so resize accordingly)
        # The seq includes BOS and all generated tokens, possibly EOS as well: trim any trailing tokens equal to padding
        mask_indices = (mask != 0).nonzero(as_tuple=False).view(-1)
        if len(mask_indices) > 0:
            last_index = mask_indices[-1].item() + 1
        else:
            last_index = seq.size(0)
        # Strip off BOS token from beginning
        seq = seq[1:last_index]
        mask = mask[1:last_index]
        assert seq.size(0) > 0

        out_sequences.append(seq)
        out_attention_masks.append(mask)
        norm_prob = real_logprob / (seq.size(0) ** length_normalization)
        out_logprobs.append(norm_prob)

    # Pad sequences to the max length of decoded sequences
    max_dec_len = max(seq.size(0) for seq in out_sequences)
    padded_seqs = []
    padded_masks = []
    for seq, mask in zip(out_sequences, out_attention_masks):
        pad_len = max_dec_len - seq.size(0)
        if pad_len > 0:
            seq = torch.cat([seq, torch.full((pad_len,), eos_token_id, device=device, dtype=torch.long)], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_len, device=device, dtype=torch.long)], dim=0)
        padded_seqs.append(seq.unsqueeze(0))         # (1, max_dec_len)
        padded_masks.append(mask.unsqueeze(0))        # (1, max_dec_len)

    decoded_input_ids = torch.cat(padded_seqs, dim=0)          # (batch_size, max_dec_len)
    decoded_attention_mask = torch.cat(padded_masks, dim=0)    # (batch_size, max_dec_len)
    normalized_log_probs = torch.tensor(out_logprobs, dtype=torch.float, device=device)  # (batch_size,)

    return decoded_input_ids, decoded_attention_mask, normalized_log_probs
