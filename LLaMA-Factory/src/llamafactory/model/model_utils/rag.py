import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# =========================
# RAG 모듈: 사전 등록/재사용
# =========================
class RAGModules(nn.Module):
    """score_adapter: ctx_similarity([B,L,Ds]) -> LM dim([B,L,Hd])"""
    def __init__(self, in_dim: int, out_dim: int, device, dtype):
        super().__init__()
        self.score_adapter = nn.Linear(in_dim, out_dim, bias=False).to(device=device, dtype=dtype)

def attach_rag_modules(model: PreTrainedModel, ctx_sim_in_dim: int, q_proj_dim: int) -> None:
    actual = model.module if hasattr(model, "module") else model
    if not hasattr(actual, "rag_modules"):
        device = next(actual.parameters()).device
        dtype = next(actual.parameters()).dtype
        actual.rag_modules = RAGModules(ctx_sim_in_dim, q_proj_dim, device, dtype)
        logger.info("[RAG] rag_modules attached (score_adapter: %d -> %d).", ctx_sim_in_dim, q_proj_dim)
    else:
        ra = actual.rag_modules
        if (ra.score_adapter.in_features != ctx_sim_in_dim) or (ra.score_adapter.out_features != q_proj_dim):
            logger.warning("[RAG] rag_modules dims changed (%d->%d vs %d->%d). Re-creating.",
                           ra.score_adapter.in_features, ra.score_adapter.out_features,
                           ctx_sim_in_dim, q_proj_dim)
            device = ra.score_adapter.weight.device
            dtype = ra.score_adapter.weight.dtype
            actual.rag_modules = RAGModules(ctx_sim_in_dim, q_proj_dim, device, dtype)

def freeze_backbone_except_rag(model: PreTrainedModel) -> None:
    """RAG 모듈(rag_projection, rag_projection_v, rag_modules.*)만 학습."""
    models_to_check = [model]
    if hasattr(model, "module"): models_to_check.append(model.module)
    if hasattr(model, "base_model"): models_to_check.append(model.base_model)

    # Ensure RAG modules are attached before freezing
    actual_model = model.module if hasattr(model, "module") else model
    if not hasattr(actual_model, "rag_modules"):
        logger.warning("[RAG] rag_modules not found during freezing! This should be attached during model patching.")
        # Try to attach with default dimensions
        try:
            attach_rag_modules(model, 1280, 2048)  # Default dimensions
        except Exception as e:
            logger.error(f"[RAG] Failed to attach rag_modules: {e}")

    for actual in models_to_check:
        logger.info(f"[RAG] Checking model: {type(actual)}")
        for _, p in actual.named_parameters():
            p.requires_grad_(False)
        patterns = ["rag_projection.weight", "rag_projection_v.weight", "rag_modules.score_adapter.weight", "visual.merger"]
        for n, p in actual.named_parameters():
            if any(pat in n for pat in patterns):
                p.requires_grad_(True)
                logger.info(f"[RAG] FORCE ENABLE: {n}")

    total_trainable, names = 0, []
    for actual in models_to_check:
        for n, p in actual.named_parameters():
            if p.requires_grad:
                total_trainable += p.numel(); names.append(n)
    logger.info("[RAG] FINAL trainable params: %d", total_trainable)
    logger.info("[RAG] FINAL trainable names: %s", names)

def freeze_backbone_except_rag_and_lm(model: PreTrainedModel) -> None:
    """RAG 모듈과 Language Model을 함께 학습."""
    models_to_check = [model]
    if hasattr(model, "module"): models_to_check.append(model.module)
    if hasattr(model, "base_model"): models_to_check.append(model.base_model)

    for actual in models_to_check:
        logger.info(f"[RAG+LM] Checking model: {type(actual)}")
        for _, p in actual.named_parameters():
            p.requires_grad_(False)
        
        # RAG 모듈 패턴
        rag_patterns = ["rag_projection.weight", "rag_projection_v.weight", "rag_modules.score_adapter.weight"]
        
        # Language Model 패턴 (vision tower와 projector 제외)
        lm_patterns = [
            "model.layers.",  # Transformer layers
            "model.norm.",    # Layer norm
            "lm_head.",       # Language modeling head
            "embed_tokens.",  # Token embeddings
        ]
        
        # Vision tower와 projector는 제외
        forbidden_patterns = [
            "vision_tower.",
            "multi_modal_projector.",
            "visual_projection.",
        ]
        
        for n, p in actual.named_parameters():
            # RAG 모듈 활성화
            if any(pat in n for pat in rag_patterns):
                p.requires_grad_(True)
                logger.info(f"[RAG+LM] RAG ENABLE: {n}")
            # Language Model 활성화 (vision 관련 제외)
            elif any(pat in n for pat in lm_patterns) and not any(forbidden in n for forbidden in forbidden_patterns):
                p.requires_grad_(True)
                logger.info(f"[RAG+LM] LM ENABLE: {n}")

    total_trainable, names = 0, []
    for actual in models_to_check:
        for n, p in actual.named_parameters():
            if p.requires_grad:
                total_trainable += p.numel(); names.append(n)
    logger.info("[RAG+LM] FINAL trainable params: %d", total_trainable)
    logger.info("[RAG+LM] FINAL trainable names: %s", names)

# =========================
# 유틸
# =========================
def save_layer_selection_info(step: int, selected_indices: torch.Tensor, scores: torch.Tensor, weights: torch.Tensor):
    try:
        log_dir = "rag_layer_logs"; os.makedirs(log_dir, exist_ok=True)
        data = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "selected_layers": selected_indices.detach().cpu().tolist(),
            "layer_scores": scores.detach().cpu().tolist(),
            "layer_weights": weights.detach().cpu().tolist(),
        }
        with open(os.path.join(log_dir, f"layer_selection_step_{step}.json"), "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save layer selection info: {e}")

def safe_extract_hidden_state(model_outputs) -> torch.Tensor:
    if hasattr(model_outputs, 'hidden_states') and model_outputs.hidden_states is not None:
        return model_outputs.hidden_states[-1]
    elif hasattr(model_outputs, 'last_hidden_state'):
        return model_outputs.last_hidden_state
    else:
        raise ValueError("No hidden states found in model outputs")

# (참고) 지금은 안 쓰지만 유지
def compute_logits_from_vision_features(model: PreTrainedModel, vision_features: torch.Tensor, layer_index: int) -> torch.Tensor:
    actual_model = model.module if hasattr(model, 'module') else model
    if vision_features.shape[-1] == 1280:
        if vision_features.dim() == 3:
            b, _, d = vision_features.shape
            vf = vision_features.reshape(-1, d)
        else:
            vf = vision_features
            b = 1
        merger = actual_model.visual.merger
        vf = vf.to(dtype=next(merger.parameters()).dtype)
        lm_embeddings_flat = merger(vf)
        if vision_features.dim() == 3:
            g = lm_embeddings_flat.shape[0] // b
            lm_embeddings = lm_embeddings_flat.view(b, g, -1)
        else:
            lm_embeddings = lm_embeddings_flat.unsqueeze(0)

    out = actual_model.model(inputs_embeds=lm_embeddings, output_hidden_states=True, use_cache=False)
    last = out.hidden_states[-1]
    return actual_model.lm_head(last[:, -1, :])

def extract_visual_features_from_model(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], List[int]]:
    """비전 인코더 중간 레이어 훅 → (ctx_similarity[B,L,Ds], ctx_full[B,L,T,Df])"""
    print(f"[RAG] extract_visual_features ENTRY", flush=True)
    batch_size = input_ids.size(0)
    logger.info(f"Model visual blocks: {len(model.visual.blocks) if hasattr(model, 'visual') and hasattr(model.visual, 'blocks') else 'None'}")
    print(f"[RAG] Setting up hooks...", flush=True)

    collected, layer_indices, hooks = [], [], []
    def _hook(_m, _inp, out):
        if isinstance(out, torch.Tensor):
            collected.append(out.detach().clone())  # 인코더 freeze 가정
            layer_indices.append(len(collected) - 1)

    if hasattr(model, 'visual') and hasattr(model.visual, 'blocks'):
        for i in range(len(model.visual.blocks)):
            hooks.append(model.visual.blocks[i].register_forward_hook(_hook))
    elif hasattr(model, 'visual') and hasattr(model.visual, 'layers'):
        for layer in model.visual.layers: hooks.append(layer.register_forward_hook(_hook))
    elif hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'layers'):
        for layer in model.vision_tower.layers: hooks.append(layer.register_forward_hook(_hook))
    elif hasattr(model, 'model') and hasattr(model.model, 'vision_tower') and hasattr(model.model.vision_tower, 'layers'):
        for layer in model.model.vision_tower.layers: hooks.append(layer.register_forward_hook(_hook))
    else:
        for name, module in model.named_modules():
            if 'vision' in name.lower() or 'visual' in name.lower():
                layers = getattr(module, 'layers', getattr(module, 'blocks', []))
                for layer in layers: hooks.append(layer.register_forward_hook(_hook))
                break

    if not hooks: raise ValueError("No visual layers found for feature extraction")
    print(f"[RAG] Registered {len(hooks)} hooks", flush=True)
    
    try:
        print(f"[RAG] Starting model forward pass...", flush=True)
        fkw = {'input_ids': input_ids, 'attention_mask': attention_mask, 'use_cache': False}
        if pixel_values is not None:
            fkw['pixel_values'] = pixel_values
            if image_grid_thw is not None: fkw['image_grid_thw'] = image_grid_thw
        print(f"[RAG] Forward kwargs prepared, calling model...", flush=True)
        _ = model(**fkw)
        print(f"[RAG] Model forward completed", flush=True)
    finally:
        for h in hooks: h.remove()

    if len(collected) == 0:
        raise ValueError("Failed to collect visual features; ensure pixel_values are provided.")

    layer_docs_similarity, layer_docs_full, final_layer_indices = [], [], []
    for i, t in enumerate(collected):
        if t.dim() == 3:   # [B, P, D]
            doc_sim  = t.float().mean(dim=1)                         # [B, D]
            doc_full = t.float().to(dtype=torch.bfloat16)            # [B, P, D]
        elif t.dim() == 2: # [P, D]
            doc_sim  = t.float().mean(dim=0).unsqueeze(0).expand(batch_size, -1)
            doc_full = t.float().unsqueeze(0).expand(batch_size, -1, -1).to(dtype=torch.bfloat16)
        else:
            raise ValueError(f"Unexpected tensor dimension: {t.shape}")
        layer_docs_similarity.append(doc_sim); layer_docs_full.append(doc_full); final_layer_indices.append(i)

    ctx_similarity = torch.stack(layer_docs_similarity, dim=1).to(input_ids.device)  # [B, L, Ds]
    ctx_full       = torch.stack(layer_docs_full,       dim=1).to(input_ids.device)  # [B, L, T, Df]
    return (ctx_similarity, ctx_full), final_layer_indices

# =========================
# RAG-Token: Training
# =========================
def rag_forward(
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
    print(f"[RAG] rag_forward ENTRY: input_ids={input_ids.shape}, labels={labels.shape if labels is not None else None}", flush=True)
    logger.info(f"[RAG] rag_forward started with batch_size={input_ids.size(0)}")
    
    assert labels is not None, "labels must be provided for rag_forward"

    # 컨텍스트 준비
    print(f"[RAG] Checking context features...", flush=True)
    if rag_context_feats is None:
        print(f"[RAG] No cached features, extracting...", flush=True)
        if not hasattr(model, '_cached_vision_features'):
            print(f"[RAG] Calling extract_visual_features_from_model...", flush=True)
            features, layer_indices = extract_visual_features_from_model(
                model=model, input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw
            )
            print(f"[RAG] Visual features extracted successfully", flush=True)
            model._cached_vision_features = features
            model._cached_layer_indices = layer_indices
        else:
            print(f"[RAG] Using cached features", flush=True)
        rag_context_feats = model._cached_vision_features
    else:
        print(f"[RAG] Using provided context features", flush=True)
    
    print(f"[RAG] Unpacking context features...", flush=True)
    ctx_similarity, ctx_full = rag_context_feats  # [B,L,Ds], [B,L,T,Df]
    print(f"[RAG] Context unpacked: sim={ctx_similarity.shape}, full={ctx_full.shape}", flush=True)
    
    B = input_ids.size(0)
    num_layers = ctx_similarity.size(1)
    top_k = min(rag_top_k if rag_top_k is not None else getattr(model, 'rag_top_k', num_layers), num_layers)
    print(f"[RAG] B={B}, num_layers={num_layers}, top_k={top_k}", flush=True)

    # === 배치별 1회: 전 레이어 merger 출력 캐시 ===
    print(f"[RAG] Starting merger caching...", flush=True)
    actual_model = model.module if hasattr(model, 'module') else model
    merger = actual_model.visual.merger
    m_dtype = next(merger.parameters()).dtype
    _, L, Tpatch, Df = ctx_full.shape
    print(f"[RAG] Merger input: B={B}, L={L}, Tpatch={Tpatch}, Df={Df}", flush=True)
    
    with torch.no_grad():
        print(f"[RAG] Reshaping for merger...", flush=True)
        flat = ctx_full.reshape(B * L * Tpatch, Df).to(dtype=m_dtype)
        print(f"[RAG] Calling merger: flat={flat.shape}", flush=True)
        lm_emb_flat = merger(flat)                                         # [(B*L*S), Hd]
        print(f"[RAG] Merger output: {lm_emb_flat.shape}", flush=True)
        
    S = lm_emb_flat.shape[0] // (B * L)
    merged_tokens_cached = lm_emb_flat.view(B, L, S, -1).to(model.dtype)   # [B, L, S, Hd]
    print(f"[RAG] Cached tokens: {merged_tokens_cached.shape}, S={S}", flush=True)

    actual = model.module if hasattr(model, 'module') else model
    total_loss = 0.0

    # Process each token in the sequence
    for t in range(labels.size(1)):
        # Skip ignore tokens - check if ANY sample in batch has valid token
        if torch.all(labels[:, t] == -100):
            continue

        # Build prefix for current step
        if t == 0:
            prefix_ids, prefix_attn = input_ids, attention_mask
            print(f"[RAG] B={B}, L={num_layers}, T={t}, Tprefix={prefix_ids.size(1)}, S={merged_tokens_cached.size(2)}", flush=True)
            logger.info(f"[RAG DEBUG] Forward step {t}: B={B}, L={num_layers}, Tprefix={prefix_ids.size(1)}")

        else:
            chunks = []
            for i in range(t):
                if torch.any(labels[:, i] != -100):
                    tok = labels[:, i:i+1].clone()
                    tok[tok == -100] = 0
                    chunks.append(tok)
            prefix_ids = torch.cat([input_ids] + chunks, dim=-1) if chunks else input_ids
            if attention_mask is not None:
                if chunks:
                    ones = torch.ones(B, len(chunks), dtype=attention_mask.dtype, device=attention_mask.device)
                    am = attention_mask
                    if am.size(0) != B: am = am.expand(B, -1)
                    am = torch.clamp(am, 0, 1).to(dtype=ones.dtype)
                    prefix_attn = torch.cat([am, ones], dim=-1)
                else:
                    prefix_attn = attention_mask.expand(B, -1) if attention_mask.size(0) != B else attention_mask
                if prefix_attn.size(1) != prefix_ids.size(1):
                    tgt = prefix_ids.size(1)
                    if prefix_attn.size(1) > tgt:
                        prefix_attn = prefix_attn[:, :tgt]
                    else:
                        pad = torch.ones(B, tgt - prefix_attn.size(1), dtype=prefix_attn.dtype, device=prefix_attn.device)
                        prefix_attn = torch.cat([prefix_attn, pad], dim=1)
            else:
                prefix_attn = None

        # === prefix 1회 forward → hidden + KV 동시 획득
        prefix_out = model(input_ids=prefix_ids, attention_mask=prefix_attn,
                           use_cache=True, output_hidden_states=True)
        hidden = safe_extract_hidden_state(prefix_out)     # [B, Tprefix, Hd]
        query  = hidden[:, -1, :]                          # [B, Hd]
        pkv    = prefix_out.past_key_values

        # 라우팅 (scores → top-k)
        q_proj = model.rag_projection(query) if hasattr(model, 'rag_projection') else query
        if not hasattr(actual, "rag_modules"):
            attach_rag_modules(model, ctx_similarity.size(-1), q_proj.size(-1))
        ctx_sim = actual.rag_modules.score_adapter(ctx_similarity.to(dtype=actual.rag_modules.score_adapter.weight.dtype))
        qn = F.normalize(q_proj, dim=-1); vn = F.normalize(ctx_sim, dim=-1)
        if qn.dtype != vn.dtype: vn = vn.to(dtype=qn.dtype)
        scores = torch.einsum('bh,bnh->bn', qn, vn)        # [B, L]
        if top_k < num_layers:
            top_scores, selected_indices = torch.topk(scores, k=top_k, dim=-1)
            weights = torch.softmax(top_scores.float(), dim=-1)
        else:
            selected_indices = torch.arange(num_layers, device=scores.device).unsqueeze(0).expand(B, -1)
            weights = torch.softmax(scores.float(), dim=-1)

        # 후보 레이어 토큰 모음 → (B*K, S, Hd)
        K  = selected_indices.size(1)
        Hd = merged_tokens_cached.size(3)
        layer_tok = torch.gather(
            merged_tokens_cached, dim=1,
            index=selected_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, S, Hd)
        )                                   # [B, K, S, Hd]
        lm_in = layer_tok.reshape(B * K, S, Hd)

        # (A) pkv가 있으면: KV 재사용 경로 (빠름)
        if pkv is not None:
            def _expand_pkv(past_kv, K):
                return tuple((k.repeat_interleave(K, dim=0), v.repeat_interleave(K, dim=0)) for (k, v) in past_kv)
            pkv_expanded = _expand_pkv(pkv, K)

            # 마스크 (prefix_attn가 있다면 그대로 확장)
            if prefix_attn is not None:
                ones = torch.ones(B, S, dtype=prefix_attn.dtype, device=prefix_attn.device)
                attn_cat = torch.cat([prefix_attn, ones], dim=-1).repeat_interleave(K, dim=0)
            else:
                attn_cat = None

            cand_out = model(
                inputs_embeds=lm_in,                # [B*K, S, Hd]
                attention_mask=attn_cat,
                past_key_values=pkv_expanded,      # ✅ prefix 조건부
                use_cache=False,
                output_hidden_states=False,
            )
            cand_last = cand_out.logits[:, -1, :]  # [B*K, V]

        # (B) pkv가 None이면: GC 호환 fallback (prefix 임베딩 + layer 토큰을 concat)
        else:
            # prefix 임베딩
            tok_emb = model.get_input_embeddings()(prefix_ids)         # [B, Tprefix, Hd]
            tok_emb = tok_emb.repeat_interleave(K, dim=0)              # [B*K, Tprefix, Hd]
            lm_in_full = torch.cat([tok_emb, lm_in], dim=1)            # [B*K, Tprefix+S, Hd]

            # 마스크
            if prefix_attn is not None:
                ones = torch.ones(B, S, dtype=prefix_attn.dtype, device=prefix_attn.device)
                attn_cat = torch.cat([prefix_attn, ones], dim=-1).repeat_interleave(K, dim=0)  # [B*K, Tprefix+S]
            else:
                attn_cat = None

            cand_out = model(
                inputs_embeds=lm_in_full,
                attention_mask=attn_cat,
                use_cache=False,
                output_hidden_states=False,
            )
            cand_last = cand_out.logits[:, -1, :]  # [B*K, V]

        cand_logits = cand_last.view(B, K, -1).float()            # [B, K, V]
        final_logits= torch.sum(cand_logits * weights.unsqueeze(-1), dim=1)  # [B, V]

        # loss
        step_logprobs = torch.log_softmax(final_logits, dim=-1)
        tgt = labels[:, t].unsqueeze(-1)
        if (tgt.max() >= step_logprobs.size(-1)) or (tgt.min() < 0):
            tgt = torch.clamp(tgt, 0, step_logprobs.size(-1) - 1)
        total_loss += (-step_logprobs.gather(dim=-1, index=tgt).squeeze(-1)).mean()

    n_valid = (labels != -100).sum().clamp_min(1)
    return {'loss': total_loss / n_valid}

# =========================
# RAG-Token: Generation
# =========================
def rag_generate(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    rag_context_feats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    rag_top_k: Optional[int] = None,
    max_new_tokens: int = 50,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
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
    B = input_ids.size(0)
    num_layers = ctx_similarity.size(1)
    top_k = min(rag_top_k if rag_top_k is not None else getattr(model, 'rag_top_k', num_layers), num_layers)

    # 전 레이어 merger 출력 캐시
    actual_model = model.module if hasattr(model, 'module') else model
    merger = actual_model.visual.merger
    m_dtype = next(merger.parameters()).dtype
    _, L, Tpatch, Df = ctx_full.shape
    with torch.no_grad():
        flat = ctx_full.reshape(B * L * Tpatch, Df).to(dtype=m_dtype)
        lm_emb_flat = merger(flat)
    S = lm_emb_flat.shape[0] // (B * L)
    merged_tokens_cached = lm_emb_flat.view(B, L, S, -1).to(model.dtype)

    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # prefix 한 번으로 hidden + KV
            prefix_out = model(input_ids=generated, attention_mask=attention_mask,
                               use_cache=True, output_hidden_states=True)
            hidden = safe_extract_hidden_state(prefix_out)
            query  = hidden[:, -1, :]
            pkv    = prefix_out.past_key_values

            # 라우팅
            q_proj = model.rag_projection(query) if hasattr(model, 'rag_projection') else query
            actual = model.module if hasattr(model, 'module') else model
            if not hasattr(actual, "rag_modules"):
                attach_rag_modules(model, ctx_similarity.size(-1), q_proj.size(-1))
            ctx_sim = actual.rag_modules.score_adapter(ctx_similarity.to(dtype=actual.rag_modules.score_adapter.weight.dtype))
            qn = F.normalize(q_proj, dim=-1); vn = F.normalize(ctx_sim, dim=-1)
            if qn.dtype != vn.dtype: vn = vn.to(dtype=qn.dtype)
            scores = torch.einsum('bh,bnh->bn', qn, vn)
            if top_k < num_layers:
                top_scores, selected_indices = torch.topk(scores, k=top_k, dim=-1)
                weights = torch.softmax(top_scores.float(), dim=-1)
            else:
                selected_indices = torch.arange(num_layers, device=scores.device).unsqueeze(0).expand(B, -1)
                weights = torch.softmax(scores.float(), dim=-1)

            # 후보 레이어 토큰 모음 → (B*K, S, Hd)
            K  = selected_indices.size(1)
            Hd = merged_tokens_cached.size(3)
            layer_tok = torch.gather(
                merged_tokens_cached, dim=1,
                index=selected_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, S, Hd)
            )
            lm_in = layer_tok.reshape(B * K, S, Hd)

            # (A) pkv가 있으면: KV 재사용 경로 (빠름)
            if pkv is not None:
                def _expand_pkv(past_kv, K):
                    return tuple((k.repeat_interleave(K, dim=0), v.repeat_interleave(K, dim=0)) for (k, v) in past_kv)
                pkv_expanded = _expand_pkv(pkv, K)

                if attention_mask is not None:
                    ones = torch.ones(B, S, dtype=attention_mask.dtype, device=attention_mask.device)
                    attn_cat = torch.cat([attention_mask, ones], dim=-1).repeat_interleave(K, dim=0)
                else:
                    attn_cat = None

                cand_out = model(
                    inputs_embeds=lm_in,
                    attention_mask=attn_cat,
                    past_key_values=pkv_expanded,
                    use_cache=False,
                    output_hidden_states=False,
                )
                cand_last = cand_out.logits[:, -1, :]  # [B*K, V]

            # (B) pkv가 None이면: GC 호환 fallback
            else:
                # prefix 임베딩
                tok_emb = model.get_input_embeddings()(generated)          # [B, Tprefix, Hd]
                tok_emb = tok_emb.repeat_interleave(K, dim=0)              # [B*K, Tprefix, Hd]
                lm_in_full = torch.cat([tok_emb, lm_in], dim=1)            # [B*K, Tprefix+S, Hd]

                # 마스크
                if attention_mask is not None:
                    ones = torch.ones(B, S, dtype=attention_mask.dtype, device=attention_mask.device)
                    attn_cat = torch.cat([attention_mask, ones], dim=-1).repeat_interleave(K, dim=0)
                else:
                    prefix_len = generated.size(1)
                    prefix_mask = torch.ones(B, prefix_len, dtype=torch.long, device=generated.device)
                    ones = torch.ones(B, S, dtype=torch.long, device=generated.device)
                    attn_cat = torch.cat([prefix_mask, ones], dim=-1).repeat_interleave(K, dim=0)

                cand_out = model(
                    inputs_embeds=lm_in_full,
                    attention_mask=attn_cat,
                    use_cache=False,
                    output_hidden_states=False,
                )
                cand_last = cand_out.logits[:, -1, :]  # [B*K, V]

            cand_logits = cand_last.view(B, K, -1).float()
            final_logits= torch.sum(cand_logits * weights.unsqueeze(-1), dim=1)

            probs = torch.softmax(final_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated  = torch.cat([generated, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
    return generated

__all__ = [
    'rag_forward',
    'rag_generate',
    'extract_visual_features_from_model',
    'compute_logits_from_vision_features',
    'attach_rag_modules',
    'freeze_backbone_except_rag',
]
