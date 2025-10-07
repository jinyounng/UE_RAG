"""Custom wrapper to run LLaMA-Factory RAG-Sequence checkpoints in VLMEvalKit."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from .qwen2_vl.model import Qwen2VLChat

try:  # qwen-vl-utils is required for vision preprocessing
    from qwen_vl_utils import process_vision_info
except ModuleNotFoundError as err:  # pragma: no cover - fail fast when dependency missing
    logging.critical("qwen_vl_utils not found. Install via `pip install qwen-vl-utils`." " Original error: %s", err)
    raise

try:
    from llamafactory.model.model_utils.rag import attach_rag_modules
    from llamafactory.model.model_utils.rag_sequence import rag_sequence_generate
except ModuleNotFoundError as err:  # pragma: no cover - ensure llama-factory is importable
    logging.critical(
        "Failed to import LLaMA-Factory RAG helpers. "
        "Add `/data3/jykim/Projects/VLM/LLaMA-Factory/src` to PYTHONPATH or install the package." \
    )
    raise


class LLFQwen2RagSequence(Qwen2VLChat):
    """Run Qwen2.5-VL checkpoints fine-tuned with LLaMA-Factory RAG-Sequence."""

    def __init__(
        self,
        model_path: str,
        rag_top_k: int = 3,
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path=model_path, max_new_tokens=max_new_tokens, **kwargs)
        self.rag_top_k = rag_top_k
        self._prepare_rag_modules()

    def _prepare_rag_modules(self) -> None:
        """Ensure score adapter exists and is placed on the right device/dtype."""

        vision_cfg = getattr(self.model.config, "vision_config", None)
        ctx_dim = getattr(vision_cfg, "hidden_size", 1280)
        q_dim = getattr(self.model.config, "hidden_size", 2048)
        attach_rag_modules(self.model, ctx_dim, q_dim)
        # Store rag_top_k on the model for downstream helpers if needed.
        setattr(self.model, "rag_top_k", self.rag_top_k)

    def generate_inner_transformers(  # type: ignore[override]
        self,
        message: list[dict[str, str]],
        dataset: Optional[str] = None,
    ) -> str:
        """Override HF inference to route generation through rag_sequence."""

        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            logging.info("Messages: %s", messages)

        text = self.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info([messages])
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")  # VLMEvalKit expects GPU inference for local checkpoints

        pixel_values = getattr(inputs, "pixel_values", None)
        image_grid_thw = getattr(inputs, "image_grid_thw", None)

        sequences, _ = rag_sequence_generate(
            model=self.model,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            rag_top_k=self.rag_top_k,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )

        generated = sequences[:, inputs.input_ids.size(1) :]
        response = self.processor.tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        if self.post_process:
            # Reuse parent post-process behaviour if required.
            response = self._post_process_boxed(response)

        if self.verbose:
            logging.info("Response: %s", response)
        return response

    def _post_process_boxed(self, response: str) -> str:
        """Mirror Qwen2VLChat boxed extraction logic for convenience."""

        segment = response.split("\\boxed{")[-1]
        depth = 1
        for idx, char in enumerate(segment):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == 0:
                return segment[:idx]
        return segment
