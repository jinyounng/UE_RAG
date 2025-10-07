import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging
import os

logger = logging.getLogger(__name__)

# Import utilities from rag.py
from .rag import (
    attach_rag_modules, 
    safe_extract_hidden_state,
    extract_visual_features_from_model
)

# =========================
# RAG-Sequence: Training
# =========================
def rag_sequence_forward(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    rag_context_feats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    rag_top_k: Optional[int] = None,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    RAG-Sequence Forward Pass (Correct Implementation):
    1. Select top-k layers based on input sequence
    2. For each selected layer, generate full sequence logits
    3. Apply correct RAG-Sequence marginalization
    
    RAG-Sequence formula: p(y|x) = Σ_z p(z|x) * Π_t p(y_t|y_{<t}, x, z)
    """
    # Note: return a dict to match trainer expectations
    
    print(f"[RAG-SEQ] Forward - input_ids: {input_ids.shape}, labels: {labels.shape if labels is not None else None}")
    
    # === 1. Context Preparation ===
    if rag_context_feats is None:
        # 매 step마다 새로 뽑기 (캐시 사용 안함)
        rag_context_feats, _ = extract_visual_features_from_model(
            model=model, input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw
        )
    
    ctx_similarity, ctx_full = rag_context_feats
    batch_size = input_ids.size(0)
    seq_len = input_ids.size(1)
    num_layers = ctx_similarity.size(1)
    top_k = min(rag_top_k if rag_top_k is not None else getattr(model, 'rag_top_k', num_layers), num_layers)
    # 학습 시에는 top_k>=2 보장 (게이트에 grad 전달)
    if model.training:
        top_k = max(2, top_k)
    
    # === 2. Query Computation and Layer Selection ===
    # LLM 동결: 쿼리 추출은 no_grad로
    with torch.no_grad():
        query_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        hidden_states = safe_extract_hidden_state(query_outputs)  # [B, T, H]
        # Use mean pooling for query representation
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            query = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        else:
            query = hidden_states.mean(dim=1)
    
    # Layer selection via similarity scoring
    actual_model = model.module if hasattr(model, 'module') else model
    # forward 내 .to(...) 호출 제거 (초기화 단계에서 1회 정렬)
    if hasattr(model, 'rag_projection'):
        query = query.to(dtype=model.rag_projection.weight.dtype)
        q_proj = model.rag_projection(query)  # 학습 대상
    else:
        q_proj = query
    
    # RAG modules should be attached during model patching
    if not hasattr(actual_model, "rag_modules"):
        logger.error("[RAG-SEQ] rag_modules not found! This should be attached during model patching.")
        raise RuntimeError("RAG modules not found - model patching failed")
    
    # Align score adapter device/dtype with its weights
    sa_w = actual_model.rag_modules.score_adapter.weight
    ctx_sim = actual_model.rag_modules.score_adapter(
        ctx_similarity.to(device=sa_w.device, dtype=sa_w.dtype)
    )
    
    qn = F.normalize(q_proj, dim=-1)
    vn = F.normalize(ctx_sim, dim=-1)
    if qn.dtype != vn.dtype:
        vn = vn.to(dtype=qn.dtype)
    scores = torch.einsum('bh,bnh->bn', qn, vn)  # [B, num_layers]
    
    # Select top-k layers
    if top_k < num_layers:
        top_scores, selected_indices = torch.topk(scores, k=top_k, dim=-1)  # [B, K]
        doc_scores = torch.softmax(top_scores.float(), dim=-1)  # [B, K] - p(z|x)
    else:
        selected_indices = torch.arange(num_layers, device=scores.device).unsqueeze(0).expand(batch_size, -1)
        doc_scores = torch.softmax(scores.float(), dim=-1)  # [B, num_layers]
    
    print(f"[RAG-SEQ] Selected top-{top_k} layers with document probabilities")
    
    # === 3. Feature Projection ===
    merger = actual_model.visual.merger
    m_param = next(merger.parameters())
    m_dtype = m_param.dtype
    m_device = m_param.device
    _, L, Tpatch, Df = ctx_full.shape

    # merger는 학습 대상 → no_grad 금지
    flat_features = ctx_full.reshape(batch_size * L * Tpatch, Df).to(device=m_device, dtype=m_dtype)
    merged_flat = merger(flat_features)

    S = merged_flat.shape[0] // (batch_size * L)
    base_device = next(actual_model.parameters()).device
    base_dtype  = next(actual_model.parameters()).dtype
    # model.dtype이 없을 수 있으므로 안전한 dtype 사용
    merged_tokens = merged_flat.view(batch_size, L, S, -1).to(device=base_device, dtype=base_dtype)  # [B, L, S, H]
    
    # === 4. Generate Logits for Each Selected Layer ===
    K = selected_indices.size(1)
    hidden_size = merged_tokens.size(3)
    
    # Extract tokens for selected layers: [B, K, S, hidden_size]
    selected_layer_tokens = torch.gather(
        merged_tokens, dim=1,
        index=selected_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, S, hidden_size)
    )
    
    # LLM 동결: 입력 임베딩은 detach로 메모리 절약
    input_embeddings = model.get_input_embeddings()(input_ids).detach()  # [B, T, H]
    
    # === 5. HF RAG-Sequence loss: sequence-level marginalization ===
    loss = None
    doc_seq_logliks: list[torch.Tensor] = []  # each [B]
    target_labels = labels[..., 1:] if labels is not None else None  # predict positions 1..T-1

    # For optional debug preview (rank0), track the best-doc logits for sample 0
    debug_best_logits = None
    best_doc_idx_sample0 = torch.argmax(doc_scores[0]).item() if doc_scores is not None else 0

    for k in range(K):
        layer_k_tokens = selected_layer_tokens[:, k, :, :]  # [B, S, hidden_size]

        # Concatenate: [layer_tokens] + [input_sequence]
        combined_embeddings = torch.cat([layer_k_tokens, input_embeddings], dim=1)  # [B, S + seq_len, hidden_size]

        if attention_mask is not None:
            layer_attention_mask = torch.ones(
                batch_size, S, dtype=attention_mask.dtype, device=attention_mask.device
            )
            combined_attention_mask = torch.cat([layer_attention_mask, attention_mask], dim=1)
        else:
            combined_attention_mask = None

        # Forward pass
        layer_outputs = model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )

        # Extract logits for the original sequence part: [B, seq_len, vocab_size]
        sequence_logits = layer_outputs.logits[:, S:, :]

        if os.getenv("RAG_DEBUG_DECODE", "0") not in ["0", "false", "False", ""] and k == best_doc_idx_sample0:
            debug_best_logits = sequence_logits.detach()

        if target_labels is not None:
            # Shift to align: logits[..., :-1, :] predict labels[..., :]
            seq_log_probs = torch.log_softmax(sequence_logits, dim=-1)  # [B, T, V]
            pred_logprobs = seq_log_probs[:, :-1, :]                     # [B, T-1, V]
            tgt = target_labels                                          # [B, T-1]
            valid = (tgt != -100)
            tgt_safe = tgt.clamp_min(0)
            token_lp = pred_logprobs.gather(dim=-1, index=tgt_safe.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
            token_lp = token_lp.masked_fill(~valid, 0.0)
            seq_lp = token_lp.sum(dim=-1)  # [B]
            doc_seq_logliks.append(seq_lp)

    if labels is not None and len(doc_seq_logliks) == K:
        # [B, K]
        seq_lp_stack = torch.stack(doc_seq_logliks, dim=1)
        doc_log = torch.log(doc_scores.clamp_min(1e-10))
        log_marginal = torch.logsumexp(seq_lp_stack + doc_log, dim=1)  # [B]
        loss = (-log_marginal).mean()
    
    print(f"[RAG-SEQ] Forward complete - loss: {loss.item() if loss is not None else 'None'}")

    # Optional debug decode: set env RAG_DEBUG_DECODE=1 to enable
    if os.getenv("RAG_DEBUG_DECODE", "0") not in ["0", "false", "False", ""] and debug_best_logits is not None:
        try:
            # Align predictions with labels (use shifted positions) from best doc
            best_logprobs = torch.log_softmax(debug_best_logits, dim=-1)
            pred_next_ids = torch.argmax(best_logprobs[:, :-1, :], dim=-1)  # [B, seq_len-1]
            tgt_ids = labels[:, 1:].clone() if labels is not None else None   # [B, seq_len-1]

            if tgt_ids is not None:
                valid_mask = (tgt_ids != -100)
                b = 0  # first sample
                if valid_mask[b].any():
                    pred_seq = pred_next_ids[b][valid_mask[b]]
                    gt_seq = tgt_ids[b][valid_mask[b]].clamp_min(0)

                    # Lazily load tokenizer once
                    if not hasattr(model, "_debug_tokenizer"):
                        try:
                            from transformers import AutoTokenizer
                            name_or_path = getattr(model.config, "_name_or_path", None) or getattr(model, "name_or_path", None)
                            model._debug_tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
                        except Exception:
                            model._debug_tokenizer = None

                    tok = getattr(model, "_debug_tokenizer", None)
                    if tok is not None:
                        pred_text = tok.decode(pred_seq.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        gt_text = tok.decode(gt_seq.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        print(f"[RAG-SEQ][DEBUG] Pred: {pred_text[:200]}...")
                        print(f"[RAG-SEQ][DEBUG] Gold: {gt_text[:200]}...")
                    else:
                        print(f"[RAG-SEQ][DEBUG] Pred IDs: {pred_seq.tolist()[:32]} ...")
                        print(f"[RAG-SEQ][DEBUG] Gold IDs: {gt_seq.tolist()[:32]} ...")
        except Exception as e:
            print(f"[RAG-SEQ][DEBUG] decode failed: {e}")
    return {
        "loss": loss if loss is not None else torch.tensor(0.0, device=input_ids.device, requires_grad=True),
        "doc_scores": doc_scores,
    }


# =========================
# RAG-Sequence: Generation
# =========================
def rag_sequence_generate(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    rag_context_feats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    rag_top_k: Optional[int] = None,
    max_new_tokens: int = 50,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 1.0,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    **kwargs
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    RAG-Sequence Generation (Correct Implementation):
    1. Select top-k layers once based on input
    2. Generate complete sequences for each layer independently
    3. Apply sequence-level marginalization for final result
    
    Returns:
        final_sequence: [B, total_len] - RAG-Sequence marginalized result
        layer_sequences: List[Tensor] - individual sequences from each layer
    """
    print(f"[RAG-SEQ] Generate start - input: {input_ids.shape}, max_new_tokens: {max_new_tokens}")
    
    if pad_token_id is None:
        pad_token_id = model.config.pad_token_id
    if eos_token_id is None:
        eos_token_id = model.config.eos_token_id
    
    batch_size = input_ids.size(0)
    device = input_ids.device
    
    # === 1. One-time layer selection ===
    if rag_context_feats is None:
        if not hasattr(model, '_cached_vision_features'):
            features, layer_indices = extract_visual_features_from_model(
                model=model, input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw
            )
            model._cached_vision_features = features
            model._cached_layer_indices = layer_indices
        rag_context_feats = model._cached_vision_features
    
    ctx_similarity, ctx_full = rag_context_feats
    num_layers = ctx_similarity.size(1)
    top_k = min(rag_top_k if rag_top_k is not None else getattr(model, 'rag_top_k', num_layers), num_layers)
    
    # Get query and select layers
    with torch.no_grad():
        query_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True
        )
        hidden_states = safe_extract_hidden_state(query_outputs)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            query = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        else:
            query = hidden_states.mean(dim=1)
    
    # Layer selection
    actual_model = model.module if hasattr(model, 'module') else model
    if hasattr(model, 'rag_projection'):
        query = query.to(dtype=model.rag_projection.weight.dtype)
        q_proj = model.rag_projection(query)
    else:
        q_proj = query
    
    # RAG modules should be attached during model patching
    if not hasattr(actual_model, "rag_modules"):
        logger.error("[RAG-SEQ] rag_modules not found in generation! This should be attached during model patching.")
        raise RuntimeError("RAG modules not found - model patching failed")
    
    sa_w = actual_model.rag_modules.score_adapter.weight
    ctx_sim = actual_model.rag_modules.score_adapter(
        ctx_similarity.to(device=sa_w.device, dtype=sa_w.dtype)
    )
    
    qn = F.normalize(q_proj, dim=-1)
    vn = F.normalize(ctx_sim, dim=-1)
    if qn.dtype != vn.dtype:
        vn = vn.to(dtype=qn.dtype)
    scores = torch.einsum('bh,bnh->bn', qn, vn)
    
    if top_k < num_layers:
        top_scores, selected_indices = torch.topk(scores, k=top_k, dim=-1)
        doc_probs = torch.softmax(top_scores.float(), dim=-1)
    else:
        selected_indices = torch.arange(num_layers, device=device).unsqueeze(0).expand(batch_size, -1)
        doc_probs = torch.softmax(scores.float(), dim=-1)
    
    print(f"[RAG-SEQ] Selected {top_k} layers with probabilities: {doc_probs[0].tolist()}")
    
    # Cache merged tokens
    merger = actual_model.visual.merger
    m_param = next(merger.parameters())
    m_dtype = m_param.dtype
    m_device = m_param.device
    _, L, Tpatch, Df = ctx_full.shape
    
    with torch.no_grad():
        flat_features = ctx_full.reshape(batch_size * L * Tpatch, Df).to(device=m_device, dtype=m_dtype)
        merged_flat = merger(flat_features)
    
    S = merged_flat.shape[0] // (batch_size * L)
    base_device = next(actual_model.parameters()).device
    base_dtype  = next(actual_model.parameters()).dtype
    merged_tokens_cache = merged_flat.view(batch_size, L, S, -1).to(device=next(actual_model.parameters()).device, dtype=base_dtype)
    
    # === 2. Generate complete sequences for each layer ===
    K = selected_indices.size(1)
    hidden_size = merged_tokens_cache.size(3)
    
    selected_tokens = torch.gather(
        merged_tokens_cache, dim=1,
        index=selected_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, S, hidden_size)
    )
    
    layer_sequences = []
    
    for k in range(K):
        print(f"[RAG-SEQ] Generating complete sequence for layer {k+1}/{K}")
        
        layer_tokens = selected_tokens[:, k, :, :]  # [B, S, hidden_size]
        
        # Generate complete sequence for this layer
        layer_sequence = _generate_complete_sequence(
            model=model,
            layer_tokens=layer_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id
        )
        
        layer_sequences.append(layer_sequence)
        print(f"[RAG-SEQ] Layer {k+1} generated sequence length: {layer_sequence.size(1)}")
    
    # === 3. RAG-Sequence marginalization ===
    print(f"[RAG-SEQ] Applying RAG-Sequence marginalization over {K} sequences")
    
    final_sequence = _rag_sequence_marginalize_generation(
        layer_sequences=layer_sequences,
        doc_probs=doc_probs,
        input_length=input_ids.size(1),
        pad_token_id=pad_token_id
    )
    
    print(f"[RAG-SEQ] Generation complete - final sequence length: {final_sequence.size(1)}")
    
    return final_sequence, layer_sequences


def _generate_complete_sequence(
    model: PreTrainedModel,
    layer_tokens: torch.Tensor,  # [B, S, hidden_size]
    input_ids: torch.Tensor,     # [B, input_len]
    attention_mask: Optional[torch.Tensor],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    pad_token_id: int,
    eos_token_id: Optional[int]
) -> torch.Tensor:
    """Generate a complete sequence for one specific layer"""
    
    batch_size = input_ids.size(0)
    device = input_ids.device
    S = layer_tokens.size(1)
    
    # Start with input tokens
    current_ids = input_ids.clone()  # [B, input_len]
    
    # Create initial attention mask
    if attention_mask is not None:
        current_attention_mask = attention_mask.clone()
    else:
        current_attention_mask = torch.ones_like(current_ids)
    
    # Track unfinished sequences
    unfinished = torch.ones(batch_size, dtype=torch.long, device=device)
    
    # Generation loop
    for step in range(max_new_tokens):
        with torch.no_grad():
            # Convert current tokens to embeddings
            current_embeddings = model.get_input_embeddings()(current_ids)  # [B, current_len, hidden_size]
            
            # Combine layer tokens + current sequence
            combined_embeddings = torch.cat([layer_tokens, current_embeddings], dim=1)  # [B, S + current_len, hidden_size]
            
            # Create combined attention mask
            layer_mask = torch.ones(batch_size, S, dtype=current_attention_mask.dtype, device=device)
            combined_mask = torch.cat([layer_mask, current_attention_mask], dim=1)
            
            # Forward pass
            outputs = model(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_mask,
                use_cache=False
            )
            
            # Get next token logits (last position)
            next_token_logits = outputs.logits[:, -1, :] / temperature  # [B, vocab_size]
            
            # Apply top-p filtering
            if do_sample and top_p < 1.0:
                next_token_logits = _top_p_filtering(next_token_logits, top_p)
            
            # Sample next token
            if do_sample:
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)  # [B]
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)  # [B]
            
            # Only update unfinished sequences
            next_tokens = next_tokens * unfinished + pad_token_id * (1 - unfinished)
            
            # Append to current sequence
            current_ids = torch.cat([current_ids, next_tokens.unsqueeze(1)], dim=-1)
            
            # Update attention mask
            new_mask = torch.ones(batch_size, 1, dtype=current_attention_mask.dtype, device=device)
            current_attention_mask = torch.cat([current_attention_mask, new_mask], dim=-1)
            
            # Check for EOS
            if eos_token_id is not None:
                unfinished = unfinished.mul((next_tokens != eos_token_id).long())
            
            # Stop if all sequences finished
            if unfinished.max() == 0:
                break
    
    return current_ids


def _rag_sequence_marginalize_generation(
    layer_sequences: List[torch.Tensor],  # List of [B, seq_len] 
    doc_probs: torch.Tensor,              # [B, K]
    input_length: int,
    pad_token_id: int
) -> torch.Tensor:
    """
    Apply RAG-Sequence marginalization for generation.
    
    In the original RAG-Sequence, we should evaluate the probability of complete sequences
    and then sample. For practical generation, we use a simplified approach that
    respects the document probabilities.
    """
    batch_size = doc_probs.size(0)
    K = len(layer_sequences)
    device = doc_probs.device
    
    # Find max length
    max_len = max(seq.size(1) for seq in layer_sequences)
    
    # Pad all sequences to same length
    padded_sequences = []
    for seq in layer_sequences:
        if seq.size(1) < max_len:
            pad_len = max_len - seq.size(1)
            padding = torch.full((batch_size, pad_len), pad_token_id, dtype=seq.dtype, device=device)
            padded_seq = torch.cat([seq, padding], dim=1)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    # Stack: [B, K, max_len]
    stacked_sequences = torch.stack(padded_sequences, dim=1)
    
    # For RAG-Sequence marginalization in generation:
    # We select the sequence with highest probability, but blend at positions of disagreement
    final_sequences = []
    
    for b in range(batch_size):
        batch_doc_probs = doc_probs[b]  # [K]
        batch_sequences = stacked_sequences[b]  # [K, max_len]
        
        # Primary strategy: use the highest probability document as base
        best_doc_idx = torch.argmax(batch_doc_probs).item()
        final_sequence = batch_sequences[best_doc_idx].clone()  # [max_len]
        
        # Secondary strategy: at positions where sequences disagree, 
        # use probabilistic blending
        for pos in range(input_length, max_len):  # Only for generated tokens
            pos_tokens = batch_sequences[:, pos]  # [K]
            
            # Skip if all padding
            if (pos_tokens == pad_token_id).all():
                break
                
            # Find valid tokens
            valid_mask = (pos_tokens != pad_token_id)
            if valid_mask.sum() > 1:  # Multiple valid options
                valid_tokens = pos_tokens[valid_mask]
                valid_probs = batch_doc_probs[valid_mask]
                
                # If there's disagreement, use probabilistic selection
                unique_tokens = torch.unique(valid_tokens)
                if len(unique_tokens) > 1:
                    # Aggregate probabilities for each unique token
                    token_probs = {}
                    for i, token in enumerate(valid_tokens):
                        token_item = token.item()
                        if token_item not in token_probs:
                            token_probs[token_item] = 0.0
                        token_probs[token_item] += valid_probs[i].item()
                    
                    # Sample based on aggregated probabilities
                    tokens = list(token_probs.keys())
                    probs = torch.tensor([token_probs[t] for t in tokens], device=device)
                    probs = probs / probs.sum()
                    
                    selected_idx = torch.multinomial(probs, num_samples=1).item()
                    final_sequence[pos] = tokens[selected_idx]
        
        final_sequences.append(final_sequence)
    
    return torch.stack(final_sequences, dim=0)


def _top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply top-p (nucleus) filtering"""
    if top_p >= 1.0:
        return logits
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    return logits


# =========================
# RAG-Sequence: Concat Last Layer with Top-K
# =========================
def concat_last_layer_with_topk(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    rag_context_feats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    rag_top_k: Optional[int] = None,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    RAG-Sequence with Last Layer Concatenation:
    1. Always include the last layer (most important)
    2. Select top-k from remaining layers
    3. Concatenate features and average them
    4. Apply RAG-Sequence marginalization
    
    This ensures the last layer is always included while still benefiting
    from diversity of other layers.
    """
    print(f"[RAG-CONCAT] Forward - input_ids: {input_ids.shape}, labels: {labels.shape if labels is not None else None}")
    
    # === 1. Context Preparation ===
    if rag_context_feats is None:
        # 매 step마다 새로 뽑기 (캐시 사용 안함)
        rag_context_feats, _ = extract_visual_features_from_model(
            model=model, input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw
        )
    
    ctx_similarity, ctx_full = rag_context_feats
    batch_size = input_ids.size(0)
    seq_len = input_ids.size(1)
    num_layers = ctx_similarity.size(1)
    top_k = min(rag_top_k if rag_top_k is not None else getattr(model, 'rag_top_k', num_layers), num_layers)
    
    # === 2. Query Computation and Layer Selection ===
    with torch.no_grad():
        query_outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            use_cache=False, 
            output_hidden_states=True
        )
        hidden_states = safe_extract_hidden_state(query_outputs)  # [B, seq_len, hidden_size]
        
        # Use mean pooling for query representation
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
            query = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)  # [B, hidden_size]
        else:
            query = hidden_states.mean(dim=1)  # [B, hidden_size]
    
    # Layer selection via similarity scoring
    actual_model = model.module if hasattr(model, 'module') else model
    
    if hasattr(model, 'rag_projection'):
        try:
            model.rag_projection.to(device=query.device)
        except Exception:
            pass
        query = query.to(dtype=model.rag_projection.weight.dtype)
        q_proj = model.rag_projection(query)
    else:
        q_proj = query
    
    # RAG modules should be attached during model patching
    if not hasattr(actual_model, "rag_modules"):
        logger.error("[RAG-CONCAT] rag_modules not found! This should be attached during model patching.")
        raise RuntimeError("RAG modules not found - model patching failed")
    
    # Align score adapter device/dtype with its weights
    sa_w = actual_model.rag_modules.score_adapter.weight
    ctx_sim = actual_model.rag_modules.score_adapter(
        ctx_similarity.to(device=sa_w.device, dtype=sa_w.dtype)
    )
    
    qn = F.normalize(q_proj, dim=-1)
    vn = F.normalize(ctx_sim, dim=-1)
    if qn.dtype != vn.dtype:
        vn = vn.to(dtype=qn.dtype)
    scores = torch.einsum('bh,bnh->bn', qn, vn)  # [B, num_layers]
    
    # === 3. Special Layer Selection: Last Layer + Top-K from Others ===
    last_layer_idx = num_layers - 1  # Last layer index
    
    if top_k <= 1:
        # Only use last layer
        selected_indices = torch.full((batch_size, 1), last_layer_idx, device=scores.device)
        doc_scores = torch.ones(batch_size, 1, device=scores.device)
        print(f"[RAG-CONCAT] Using only last layer (index {last_layer_idx})")
    else:
        # Use last layer + top-(k-1) from remaining layers
        remaining_layers = torch.arange(num_layers - 1, device=scores.device)  # Exclude last layer
        remaining_scores = scores[:, :-1]  # [B, num_layers-1]
        
        # Select top-1 from remaining layers (always just 1 additional layer)
        if top_k - 1 > 0:
            top_remaining_scores, top_remaining_indices = torch.topk(
                remaining_scores, k=1, dim=-1
            )  # [B, 1]
            
            # Combine last layer with selected remaining layers
            last_layer_indices = torch.full((batch_size, 1), last_layer_idx, device=scores.device)
            selected_indices = torch.cat([last_layer_indices, top_remaining_indices], dim=-1)  # [B, k]
            
            # Combine scores: last layer score + top remaining scores
            last_layer_scores = scores[:, last_layer_idx:last_layer_idx+1]  # [B, 1]
            combined_scores = torch.cat([last_layer_scores, top_remaining_scores], dim=-1)  # [B, k]
            doc_scores = torch.softmax(combined_scores.float(), dim=-1)  # [B, k]
        else:
            # Only last layer
            selected_indices = torch.full((batch_size, 1), last_layer_idx, device=scores.device)
            doc_scores = torch.ones(batch_size, 1, device=scores.device)
        
        print(f"[RAG-CONCAT] Selected layers: {selected_indices[0].tolist()} (last layer + 1 from others)")
    
    # === 4. Feature Projection ===
    merger = actual_model.visual.merger
    m_param = next(merger.parameters())
    m_dtype = m_param.dtype
    m_device = m_param.device
    _, L, Tpatch, Df = ctx_full.shape
    
    with torch.no_grad():
        flat_features = ctx_full.reshape(batch_size * L * Tpatch, Df).to(device=m_device, dtype=m_dtype)
        merged_flat = merger(flat_features)
    
    S = merged_flat.shape[0] // (batch_size * L)
    merged_tokens_cache = merged_flat.view(batch_size, L, S, -1).to(device=next(actual_model.parameters()).device, dtype=model.dtype)  # [B, L, S, hidden_size]
    
    # === 5. Generate Logits for Each Selected Layer ===
    K = selected_indices.size(1)
    hidden_size = merged_tokens_cache.size(3)
    
    # Extract tokens for selected layers: [B, K, S, hidden_size]
    selected_layer_tokens = torch.gather(
        merged_tokens_cache, dim=1,
        index=selected_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, S, hidden_size)
    )
    
    input_embeddings = model.get_input_embeddings()(input_ids)  # [B, seq_len, hidden_size]
    
    # === 6. HF RAG-Sequence loss: sequence-level marginalization ===
    loss = None
    doc_seq_logliks: list[torch.Tensor] = []  # each [B]
    target_labels = labels[..., 1:] if labels is not None else None  # predict positions 1..T-1

    # For optional debug preview (rank0), track the best-doc logits for sample 0
    debug_best_logits = None
    best_doc_idx_sample0 = torch.argmax(doc_scores[0]).item() if doc_scores is not None else 0

    for k in range(K):
        layer_k_tokens = selected_layer_tokens[:, k, :, :]  # [B, S, hidden_size]

        # Concatenate: [layer_tokens] + [input_sequence]
        combined_embeddings = torch.cat([layer_k_tokens, input_embeddings], dim=1)  # [B, S + seq_len, hidden_size]

        if attention_mask is not None:
            layer_attention_mask = torch.ones(
                batch_size, S, dtype=attention_mask.dtype, device=attention_mask.device
            )
            combined_attention_mask = torch.cat([layer_attention_mask, attention_mask], dim=1)
        else:
            combined_attention_mask = None

        # Forward pass
        layer_outputs = model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )

        # Extract logits for the original sequence part: [B, seq_len, vocab_size]
        sequence_logits = layer_outputs.logits[:, S:, :]

        if os.getenv("RAG_DEBUG_DECODE", "0") not in ["0", "false", "False", ""] and k == best_doc_idx_sample0:
            debug_best_logits = sequence_logits.detach()

        if target_labels is not None:
            # Shift to align: logits[..., :-1, :] predict labels[..., :]
            seq_log_probs = torch.log_softmax(sequence_logits, dim=-1)  # [B, T, V]
            pred_logprobs = seq_log_probs[:, :-1, :]                     # [B, T-1, V]
            tgt = target_labels                                          # [B, T-1]
            valid = (tgt != -100)
            tgt_safe = tgt.clamp_min(0)
            token_lp = pred_logprobs.gather(dim=-1, index=tgt_safe.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
            token_lp = token_lp.masked_fill(~valid, 0.0)
            seq_lp = token_lp.sum(dim=-1)  # [B]
            doc_seq_logliks.append(seq_lp)

    if labels is not None and len(doc_seq_logliks) == K:
        # [B, K]
        seq_lp_stack = torch.stack(doc_seq_logliks, dim=1)
        doc_log = torch.log(doc_scores.clamp_min(1e-10))
        log_marginal = torch.logsumexp(seq_lp_stack + doc_log, dim=1)  # [B]
        loss = (-log_marginal).mean()
    
    print(f"[RAG-CONCAT] Forward complete - loss: {loss.item() if loss is not None else 'None'}")

    # Optional debug decode: set env RAG_DEBUG_DECODE=1 to enable
    if os.getenv("RAG_DEBUG_DECODE", "0") not in ["0", "false", "False", ""] and debug_best_logits is not None:
        try:
            # Align predictions with labels (use shifted positions) from best doc
            best_logprobs = torch.log_softmax(debug_best_logits, dim=-1)
            pred_next_ids = torch.argmax(best_logprobs[:, :-1, :], dim=-1)  # [B, seq_len-1]
            tgt_ids = labels[:, 1:].clone() if labels is not None else None   # [B, seq_len-1]

            if tgt_ids is not None:
                valid_mask = (tgt_ids != -100)
                b = 0  # first sample
                if valid_mask[b].any():
                    pred_seq = pred_next_ids[b][valid_mask[b]]
                    gt_seq = tgt_ids[b][valid_mask[b]].clamp_min(0)

                    # Lazily load tokenizer once
                    if not hasattr(model, "_debug_tokenizer"):
                        try:
                            from transformers import AutoTokenizer
                            name_or_path = getattr(model.config, "_name_or_path", None) or getattr(model, "name_or_path", None)
                            model._debug_tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
                        except Exception:
                            model._debug_tokenizer = None

                    tok = getattr(model, "_debug_tokenizer", None)
                    if tok is not None:
                        pred_text = tok.decode(pred_seq.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        gt_text = tok.decode(gt_seq.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        print(f"[RAG-CONCAT][DEBUG] Pred: {pred_text[:200]}...")
                        print(f"[RAG-CONCAT][DEBUG] Gold: {gt_text[:200]}...")
                    else:
                        print(f"[RAG-CONCAT][DEBUG] Pred IDs: {pred_seq.tolist()[:32]} ...")
                        print(f"[RAG-CONCAT][DEBUG] Gold IDs: {gt_seq.tolist()[:32]} ...")
        except Exception as e:
            print(f"[RAG-CONCAT][DEBUG] decode failed: {e}")
    
    return {
        "loss": loss if loss is not None else torch.tensor(0.0, device=input_ids.device, requires_grad=True),
        "doc_scores": doc_scores,
    }


__all__ = [
    'rag_sequence_forward',
    'rag_sequence_generate',
    'concat_last_layer_with_topk',
]
