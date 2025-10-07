# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from peft import PeftModel
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..extras import logging
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_transformers_version_greater_than
from .model_utils.attention import configure_attn_implementation, print_attn_implementation
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.embedding import resize_embedding_layer
from .model_utils.kv_cache import configure_kv_cache
from .model_utils.longlora import configure_longlora
from .model_utils.moe import add_z3_leaf_module, configure_moe
from .model_utils.packing import configure_packing
from .model_utils.quantization import configure_quantization
from .model_utils.rope import configure_rope
from .model_utils.valuehead import prepare_valuehead_model
from .model_utils.visual import autocast_projector_dtype, configure_visual_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import ModelArguments


logger = logging.get_logger(__name__)


def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if model_args.model_max_length is not None and tokenizer.model_max_length < model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length  # enlarge the tokenizer max length

    if model_args.add_tokens is not None:
        num_added_tokens = tokenizer.add_tokens(new_tokens=model_args.add_tokens, special_tokens=False)
        logger.info_rank0("Add tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New tokens have been added, changed `resize_vocab` to True.")

    if model_args.add_special_tokens is not None:
        num_added_special_tokens = tokenizer.add_tokens(new_tokens=model_args.add_special_tokens, special_tokens=True)
        logger.info_rank0(
            "Add special tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_special_tokens))
        )
        if num_added_special_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New special tokens have been added, changed `resize_vocab` to True.")


def patch_processor(
    processor: "ProcessorMixin",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
) -> None:
    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_max_pixels", model_args.image_max_pixels)
    setattr(processor, "image_min_pixels", model_args.image_min_pixels)
    setattr(processor, "image_do_pan_and_scan", model_args.image_do_pan_and_scan)
    setattr(processor, "crop_to_patches", model_args.crop_to_patches)
    setattr(processor, "video_max_pixels", model_args.video_max_pixels)
    setattr(processor, "video_min_pixels", model_args.video_min_pixels)
    setattr(processor, "video_fps", model_args.video_fps)
    setattr(processor, "video_maxlen", model_args.video_maxlen)
    setattr(processor, "use_audio_in_video", model_args.use_audio_in_video)
    setattr(processor, "audio_sampling_rate", model_args.audio_sampling_rate)


def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: dict[str, Any],
    is_trainable: bool,
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    configure_attn_implementation(config, model_args)
    configure_rope(config, model_args)
    configure_longlora(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
    configure_packing(model_args, is_trainable)
    configure_kv_cache(config, model_args, is_trainable)

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    if getattr(config, "model_type", None) == "minicpmo":
        setattr(config, "init_audio", True)
        setattr(config, "init_tts", False)

    # replace the top-k gating method
    if getattr(config, "model_type", None) == "kimi_vl" and is_trainable:
        setattr(config.text_config, "topk_method", "greedy")

    if "InternVLChatModel" in getattr(config, "architectures", []):
        raise ValueError(
            "Please download the internvl models in a Hugging Face–compatible format "
            "(for example, https://huggingface.co/OpenGVLab/InternVL3-8B-hf)."
        )

    if "LlavaLlamaForCausalLM" in getattr(config, "architectures", []):
        raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")

    if getattr(config, "model_type", None) == "internlm3" and not is_transformers_version_greater_than("4.47.1"):
        raise RuntimeError("InternLM3 model requires transformers>=4.47.1, please upgrade it.")

    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())

    # do not cast data type of the model deepspeed zero3 without qlora
    if not (is_deepspeed_zero3_enabled() and model_args.quantization_bit is None):
        init_kwargs["torch_dtype"] = model_args.compute_dtype

        if init_kwargs["low_cpu_mem_usage"] and not is_fsdp_enabled():  # fsdp does not need device map
            if "device_map" not in init_kwargs and model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map  # device map requires low_cpu_mem_usage=True

            if init_kwargs.get("device_map", None) == "auto":
                init_kwargs["offload_folder"] = model_args.offload_folder


def patch_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    gen_config = model.generation_config  # check and fix generation config
    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    if getattr(model.config, "model_type", None) not in ["minicpmv", "minicpmo"] and "GenerationMixin" not in str(
        model.generate.__func__
    ):
        model.generate = MethodType(GenerationMixin.generate, model)

    if add_valuehead:
        prepare_valuehead_model(model)

    if model_args.resize_vocab:
        resize_embedding_layer(model, tokenizer)

        # RAG setup after model preparation to ensure it persists
    if getattr(model_args, "enable_rag", False):
        patch_model_for_rag(model, model_args)

    if is_trainable:
        if getattr(model.config, "model_type", None) == "gemma3n":
            setattr(model_args, "disable_gradient_checkpointing", True)

        prepare_model_for_training(model, model_args)
        autocast_projector_dtype(model, model_args)
        add_z3_leaf_module(model)

        
        # Re-apply RAG settings after any model wrapping (e.g., PeftModel)
        if hasattr(model, "base_model"):
            # Copy RAG attributes to base_model
            if hasattr(model, "rag_projection"):
                model.base_model.rag_projection = model.rag_projection
            if hasattr(model, "rag_projection_v"):
                model.base_model.rag_projection_v = model.rag_projection_v
            if hasattr(model, "rag_forward"):
                model.base_model.rag_forward = model.rag_forward
            if hasattr(model, "compute_logits_from_vision_features"):
                model.base_model.compute_logits_from_vision_features = model.compute_logits_from_vision_features
            # NEW: rag_modules까지 복사 (score_adapter가 Optimizer에 잡히도록)
            if hasattr(model, "rag_modules"):
                model.base_model.rag_modules = model.rag_modules

            model.base_model.enable_rag = getattr(model, "enable_rag", False)
            model.base_model.rag_top_k = getattr(model, "rag_top_k", None)

        # NEW: 래핑 이후에도 RAG 설정에 따라 동결 적용
        from .model_utils.rag import freeze_backbone_except_rag as _freeze_rag_only, freeze_backbone_except_rag_and_lm as _freeze_rag_and_lm
        if getattr(model_args, "enable_rag", False):
            logger.info_rank0("=== FINAL RAG FREEZE: Re-enabling RAG modules after all other freezing ===")
            
            rag_training_mode = getattr(model_args, 'rag_training_mode', 'auto')
            
            if rag_training_mode == 'rag_only':
                _freeze_rag_only(model)
                logger.info_rank0("Final freeze: RAG-only mode (explicit)")
            elif rag_training_mode == 'rag_lm_joint':
                _freeze_rag_and_lm(model)
                logger.info_rank0("Final freeze: RAG + LM joint mode (explicit)")
            elif rag_training_mode == 'auto':
                if getattr(model_args, 'freeze_language_model', True):
                    _freeze_rag_only(model)
                    logger.info_rank0("Final freeze: RAG-only mode (auto: freeze_language_model=True)")
                else:
                    _freeze_rag_and_lm(model)
                    logger.info_rank0("Final freeze: RAG + LM joint mode (auto: freeze_language_model=False)")
            else:
                raise ValueError(f"Invalid rag_training_mode: {rag_training_mode}")

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)

    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning_rank0("Cannot properly tag the model.")


def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)
    setattr(model, "tie_weights", MethodType(tie_weights, model))
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    setattr(model, "get_output_embeddings", MethodType(get_output_embeddings, model))
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))


def _infer_ctx_sim_in_dim(model: "PreTrainedModel", fallback: int = 1280) -> int:
    """레이어 요약 임베딩(ctx_similarity)의 마지막 차원(Ds)을 안전하게 추론."""
    m = model.base_model if hasattr(model, "base_model") else model
    # Qwen/InternVL/CLIP계열
    for path in [
        ("visual", "config", "hidden_size"),
        ("visual", "vision_model", "config", "hidden_size"),
        ("vision_tower", "config", "hidden_size"),
        ("model", "vision_tower", "config", "hidden_size"),
        ("vision_model", "config", "hidden_size"),
    ]:
        cur = m
        ok = True
        for key in path:
            if hasattr(cur, key):
                cur = getattr(cur, key)
            else:
                ok = False
                break
        if ok and isinstance(cur, int):
            return cur
    # 못 찾으면 폴백(네 파이프라인이 Qwen-VL이면 1280이 보통 맞음)
    return fallback


def patch_model_for_rag(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    """Add RAG infrastructure: projections and configuration."""
    logger.info_rank0("=== PATCH_MODEL_FOR_RAG CALLED ===")
    logger.info_rank0(f"Model type: {type(model)}")
    logger.info_rank0(f"enable_rag in model_args: {getattr(model_args, 'enable_rag', 'NOT_FOUND')}")
    
    # Determine hidden size safely
    if hasattr(model.config, 'hidden_size'):
        hidden_size = model.config.hidden_size
    elif hasattr(model.config, 'd_model'):
        hidden_size = model.config.d_model
    else:
        raise ValueError("Cannot determine hidden size from model.config")
    
    # Add RAG projection layers if missing and place on the same device/dtype as the model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if not hasattr(model, 'rag_projection'):
        proj = torch.nn.Linear(hidden_size, hidden_size, bias=False).to(device=device, dtype=dtype)
        model.add_module('rag_projection', proj)
    else:
        # ensure placement
        model.rag_projection.to(device=device, dtype=dtype)
    if not hasattr(model, 'rag_projection_v'):
        proj_v = torch.nn.Linear(hidden_size, hidden_size, bias=False).to(device=device, dtype=dtype)
        model.add_module('rag_projection_v', proj_v)
    else:
        model.rag_projection_v.to(device=device, dtype=dtype)
    
    # Store RAG configuration
    setattr(model, 'rag_top_k', getattr(model_args, 'rag_top_k', None))
    setattr(model, 'enable_rag', True)
    setattr(model, 'rag_layer_selection_mode', getattr(model_args, 'rag_layer_selection_mode', 'standard'))
    
    # Mark RAG modules as trainable (not affected by LoRA)
    if hasattr(model, 'rag_projection'):
        for param in model.rag_projection.parameters():
            param.requires_grad = True
    if hasattr(model, 'rag_projection_v'):
        for param in model.rag_projection_v.parameters():
            param.requires_grad = True
    
    # Bind RAG methods
    from .model_utils.rag import compute_logits_from_vision_features, attach_rag_modules, freeze_backbone_except_rag
    from .model_utils.rag_sequence import rag_sequence_forward, concat_last_layer_with_topk
    
    # Select RAG function based on configuration
    rag_layer_selection_mode = getattr(model_args, 'rag_layer_selection_mode', 'standard')
    
    if rag_layer_selection_mode == 'concat_last_layer':
        model.rag_forward = MethodType(concat_last_layer_with_topk, model)
        logger.info_rank0("RAG mode: concat_last_layer_with_topk")
    else:
        model.rag_forward = MethodType(rag_sequence_forward, model)
        logger.info_rank0("RAG mode: standard rag_sequence_forward")
    
    model.compute_logits_from_vision_features = MethodType(compute_logits_from_vision_features, model)
    
    # ===== RAG 모듈 사전 등록 (Optimizer 만들기 전) =====
    lm_dim = model.get_input_embeddings().embedding_dim
    ctx_sim_in_dim = _infer_ctx_sim_in_dim(model, fallback=1280)  # ← 모델에서 자동 추론
    attach_rag_modules(model, ctx_sim_in_dim=ctx_sim_in_dim, q_proj_dim=lm_dim)

    # 1차 동결: RAG 훈련 모드에 따라 결정
    rag_training_mode = getattr(model_args, 'rag_training_mode', 'auto')
    
    if rag_training_mode == 'rag_only':
        # RAG 모듈만 훈련
        freeze_backbone_except_rag(model)
        logger.info_rank0("RAG-only training mode enabled (explicit)")
    elif rag_training_mode == 'rag_lm_joint':
        # RAG + LM 동시 훈련
        from .model_utils.rag import freeze_backbone_except_rag_and_lm
        freeze_backbone_except_rag_and_lm(model)
        logger.info_rank0("RAG + LM joint training mode enabled (explicit)")
    elif rag_training_mode == 'auto':
        # freeze_language_model 설정에 따라 자동 결정
        if getattr(model_args, 'freeze_language_model', True):
            freeze_backbone_except_rag(model)
            logger.info_rank0("RAG-only training mode enabled (auto: freeze_language_model=True)")
        else:
            from .model_utils.rag import freeze_backbone_except_rag_and_lm
            freeze_backbone_except_rag_and_lm(model)
            logger.info_rank0("RAG + LM joint training mode enabled (auto: freeze_language_model=False)")
    else:
        raise ValueError(f"Invalid rag_training_mode: {rag_training_mode}. Must be one of ['rag_only', 'rag_lm_joint', 'auto']")
    
    logger.info_rank0(f"RAG infrastructure added (hidden={hidden_size}, top_k={getattr(model, 'rag_top_k', None)})")
    
    # Debug: 훈련 가능한 파라미터 확인
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.info_rank0(f"Trainable parameters: {trainable_params}")
