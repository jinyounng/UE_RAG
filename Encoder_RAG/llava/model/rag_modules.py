from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionRAGScoreAdapter(nn.Module):
    """Projects per-layer vision hidden states into the language model space for RAG."""

    def __init__(self, input_dim: int, output_dim: int, select_feature: str = 'patch', pool_mode: str = 'mean'):
        super().__init__()
        self.select_feature = select_feature
        self.pool_mode = pool_mode
        self.proj = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: sequence of tensors or a tensor containing per-layer outputs.
                Accepts:
                    - tuple/list where each element is [B, T, input_dim]
                    - tensor shaped [B, L, T, input_dim]

        Returns:
            torch.Tensor: [B, L, output_dim] projected representations.
        """

        if hidden_states is None:
            raise ValueError("hidden_states is required for VisionRAGScoreAdapter")

        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.dim() != 4:
                raise ValueError(f"Expected hidden_states tensor with 4 dims, got {hidden_states.shape}")
            layers = [hidden_states[:, idx] for idx in range(hidden_states.size(1))]
        else:
            layers = list(hidden_states)

        pooled = []
        for layer_tensor in layers:
            if not isinstance(layer_tensor, torch.Tensor):
                continue
            if layer_tensor.dim() != 3:
                continue
            tokens = self._select_tokens(layer_tensor)
            pooled.append(self._pool(tokens))

        if not pooled:
            raise ValueError("No valid layer tensors provided to VisionRAGScoreAdapter")

        stacked = torch.stack(pooled, dim=1)
        weight = self.proj.weight
        stacked = stacked.to(device=weight.device, dtype=weight.dtype)
        return self.proj(stacked)

    def _select_tokens(self, layer_tensor: torch.Tensor) -> torch.Tensor:
        if self.select_feature == 'patch':
            if layer_tensor.size(1) > 1:
                return layer_tensor[:, 1:]
            return layer_tensor
        if self.select_feature in ('cls_patch', 'cls'):
            return layer_tensor
        raise ValueError(f"Unexpected select_feature: {self.select_feature}")

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.size(1) == 0:
            raise ValueError("Token sequence is empty during pooling")
        if self.pool_mode == 'mean':
            return tokens.mean(dim=1)
        if self.pool_mode == 'cls_first':
            return tokens[:, 0]
        raise ValueError(f"Unsupported pool_mode: {self.pool_mode}")


def build_vision_rag_scorer(vision_hidden_size: int, language_hidden_size: int, select_feature: str = 'patch', pool_mode: str = 'mean'):
    """Factory helper that mirrors other vision modules."""
    return VisionRAGScoreAdapter(
        input_dim=vision_hidden_size,
        output_dim=language_hidden_size,
        select_feature=select_feature,
        pool_mode=pool_mode,
    )


def normalize_for_similarity(tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize along the last dimension for cosine similarity."""
    return F.normalize(tensor, dim=-1, eps=eps)


def select_topk_layers(
    query: torch.Tensor,
    layer_reprs: torch.Tensor,
    top_k: int,
    temperature: float = 1.0,
    return_scores: bool = False,
):
    """
    Compute cosine similarity between query and per-layer representations and return top-k.

    Args:
        query: [B, Hd] tensor representing the prompt embedding.
        layer_reprs: [B, L, Hd] tensor from VisionRAGScoreAdapter.
        top_k: number of layers to select.
        temperature: softmax temperature for weighting.
        return_scores: when True, also return the full cosine similarity scores.

    Returns:
        selected_indices: [B, K]
        layer_weights:   [B, K]
        scores (optional): [B, L]
    """

    if query.dim() != 2:
        raise ValueError(f"query must be [B, Hd], got {query.shape}")
    if layer_reprs.dim() != 3:
        raise ValueError(f"layer_reprs must be [B, L, Hd], got {layer_reprs.shape}")

    B, L, Hd = layer_reprs.shape
    if query.size(0) != B or query.size(1) != Hd:
        raise ValueError("query and layer_reprs must share batch size and hidden dim")

    top_k = max(1, min(top_k, L))

    qn = normalize_for_similarity(query)
    vn = normalize_for_similarity(layer_reprs)
    scores = torch.einsum('bh,blh->bl', qn, vn)

    top_scores, selected_indices = torch.topk(scores, k=top_k, dim=-1)
    if temperature != 1.0:
        top_scores = top_scores / temperature
    layer_weights = torch.softmax(top_scores.float(), dim=-1)

    if return_scores:
        return selected_indices, layer_weights, scores
    return selected_indices, layer_weights


def stack_hidden_states(hidden_states):
    """Stack a tuple/list of [B, T, D] into [B, L, T, D] tensor."""
    if isinstance(hidden_states, torch.Tensor):
        if hidden_states.dim() != 4:
            raise ValueError(f"Expected hidden_states with 4 dims, got {hidden_states.shape}")
        return hidden_states
    layers = [layer for layer in hidden_states if isinstance(layer, torch.Tensor)]
    if not layers:
        raise ValueError("No tensor elements found in hidden_states")
    return torch.stack(layers, dim=1)


def gather_topk_hidden_states(hidden_states, selected_indices):
    """Collect top-k layer hidden states based on indices.

    Args:
        hidden_states: tuple/list of [B, T, D] or stacked tensor [B, L, T, D].
        selected_indices: [B, K] tensor from select_topk_layers.

    Returns:
        torch.Tensor: [B, K, T, D]
    """

    stacked = stack_hidden_states(hidden_states)
    if stacked.dim() != 4:
        raise ValueError(f"Stacked hidden states must be [B, L, T, D], got {stacked.shape}")

    B, L, T, D = stacked.shape
    if selected_indices.size(0) != B:
        raise ValueError("Batch size mismatch between hidden_states and selected_indices")

    index = selected_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, D)
    return torch.gather(stacked, dim=1, index=index)


def compute_topk_vision_outputs(
    hidden_states,
    rag_scorer: VisionRAGScoreAdapter,
    query: torch.Tensor,
    top_k: int,
    temperature: float = 1.0,
    projector: Optional[nn.Module] = None,
    return_scores: bool = False,
):
    """Convenience wrapper that combines scoring, selection, and optional projection.

    Args:
        hidden_states: tuple/list of [B, T, D] or tensor [B, L, T, D] from vision encoder.
        rag_scorer: VisionRAGScoreAdapter instance.
        query: [B, Hd] tensor summarizing the text prompt.
        top_k: number of layers to select.
        temperature: softmax temperature for weighting.
        projector: optional multimodal projector; when provided returns projected features.
        return_scores: include full cosine scores in the result dict.

    Returns:
        dict with keys:
            layer_reprs: [B, L, Hd]
            selected_indices: [B, K]
            layer_weights: [B, K]
            selected_hidden_states: [B, K, T, D]
            projected_hidden_states: [B, K, T, H] (when projector given)
            scores: [B, L] (only if return_scores)
    """

    if rag_scorer is None:
        raise ValueError("rag_scorer must be provided")

    layer_reprs = rag_scorer(hidden_states)
    sel = select_topk_layers(
        query=query,
        layer_reprs=layer_reprs,
        top_k=top_k,
        temperature=temperature,
        return_scores=return_scores,
    )

    if return_scores:
        selected_indices, layer_weights, scores = sel
    else:
        selected_indices, layer_weights = sel
        scores = None

    selected_hidden = gather_topk_hidden_states(hidden_states, selected_indices)

    projected_hidden = None
    if projector is not None:
        B, K, T, D = selected_hidden.shape
        flat = selected_hidden.reshape(B * K * T, D)
        projected_flat = projector(flat)
        projected_hidden = projected_flat.view(B, K, T, -1)

    result = {
        'layer_reprs': layer_reprs,
        'selected_indices': selected_indices,
        'layer_weights': layer_weights,
        'selected_hidden_states': selected_hidden,
        'projected_hidden_states': projected_hidden,
    }

    if scores is not None:
        result['scores'] = scores

    return result
