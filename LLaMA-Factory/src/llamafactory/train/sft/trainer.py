# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        # Check if RAG is enabled

        if getattr(model, 'enable_rag', False):
            # Use RAG-Sequence forward for training
            
            # Extract inputs
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            labels = inputs.get("labels")
            pixel_values = inputs.get("pixel_values", None)  # Use pixel_values, default to None
            image_grid_thw = inputs.get("image_grid_thw", None)  # Add image_grid_thw
            
            # Get RAG parameters
            rag_top_k = getattr(model, 'rag_top_k', None)
            if rag_top_k is None:
                rag_top_k = 5  # default value
            
            # Get RAG layer selection mode from model args
            rag_layer_selection_mode = getattr(model, 'rag_layer_selection_mode', 'standard')
            
            if rag_layer_selection_mode == 'concat_last_layer':
                # Use concat_last_layer_with_topk function
                from ...model.model_utils.rag_sequence import concat_last_layer_with_topk
                rag_output = concat_last_layer_with_topk(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    rag_top_k=rag_top_k,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw
                )
            else:
                # Use standard rag_sequence_forward
                from ...model.model_utils.rag_sequence import rag_sequence_forward
                rag_output = rag_sequence_forward(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    rag_top_k=rag_top_k,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw
                )
            
            return rag_output['loss']
        else:
            # Use default compute_loss
            return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def training_step(self, model: "torch.nn.Module", inputs: Dict[str, torch.Tensor], num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """Override training_step to check gradients after backward pass"""
        # Call parent training_step
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # === GRADIENT CHECK: DeepSpeed 환경에서 RAG 모듈들 확인 ===
        # if model.training and loss is not None:
        #     print(f"[TRAINER] === GRADIENT CHECK (DeepSpeed ZeRO Stage 2) ===")
        #     
        #     # ZeRO Stage 2에서 gradient norm 확인
        #     if hasattr(self, 'deepspeed') and self.deepspeed is not None:
        #         try:
        #             # DeepSpeed의 내부 gradient norm 확인
        #             global_grad_norm = self.deepspeed.get_global_grad_norm()
        #             print(f"[TRAINER] DeepSpeed global gradient norm: {global_grad_norm:.6f}")
        #             
        #             # DeepSpeed의 내부 상태 확인
        #             print(f"[TRAINER] DeepSpeed engine: {type(self.deepspeed).__name__}")
        #             
        #         except Exception as e:
        #             print(f"[TRAINER] Error getting DeepSpeed gradient norm: {e}")
        #             
        #             # 대안: 직접 계산 시도 (ZeRO Stage 2에서는 대부분 실패)
        #             try:
        #                 total_norm = 0.0
        #                 param_count = 0
        #                 
        #                 for name, param in model.named_parameters():
        #                     if param.grad is not None and param.requires_grad:
        #                         param_norm = param.grad.data.norm(2)
        #                         total_norm += param_norm.item() ** 2
        #                         param_count += 1
        #                 
        #                 if param_count > 0:
        #                     total_norm = total_norm ** (1. / 2)
        #                     print(f"[TRAINER] Direct gradient norm: {total_norm:.6f} (from {param_count} params)")
        #                 else:
        #                     print(f"[TRAINER] No gradients found (ZeRO Stage 2 - normal)")
        #                     
        #             except Exception as e2:
        #                 print(f"[TRAINER] Direct gradient norm also failed: {e2}")
        #     else:
        #         print(f"[TRAINER] No DeepSpeed engine found")
        #     
        #     # RAG 모듈들의 requires_grad 상태 확인
        #     actual_model = model.module if hasattr(model, 'module') else model
        #     rag_modules = ["visual.merger", "rag_projection", "rag_modules.score_adapter"]
        #     
        #     for module_name in rag_modules:
        #         try:
        #             # 모듈 찾기
        #             if module_name == "visual.merger":
        #                 module = actual_model.visual.merger
        #             elif module_name == "rag_projection":
        #                 module = getattr(actual_model, 'rag_projection', None)
        #             elif module_name == "rag_modules.score_adapter":
        #                 module = actual_model.rag_modules.score_adapter
        #             else:
        #                 continue
        #             
        #             if module is None:
        #                 continue
        #             
        #             # 파라미터들의 requires_grad 상태만 체크 (ZeRO Stage 2에서는 grad 접근 불가)
        #             trainable_params = 0
        #             for name, param in module.named_parameters():
        #                 if param.requires_grad:
        #                     trainable_params += 1
        #                     print(f"[TRAINER] {name}: requires_grad=True (ZeRO Stage 2)")
        #                 else:
        #                     print(f"[TRAINER] {name}: requires_grad=False")
        #             
        #             print(f"[TRAINER] {module_name}: {trainable_params} trainable parameters")
        #                 
        #         except Exception as e:
        #             print(f"[TRAINER] Error checking {module_name}: {e}")
        #     
        #     print(f"[TRAINER] === END GRADIENT CHECK ===")
        
        return loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
