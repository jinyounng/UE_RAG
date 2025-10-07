#!/usr/bin/env python3
"""Run SpatialMQA benchmark with the latest RAG-Sequence checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import torch

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - tqdm optional
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:  # type: ignore
        return iterable

# Ensure LLaMA-Factory Python modules are importable when running from repo root.
SCRIPT_DIR = Path(__file__).resolve().parent
LLF_DIR = SCRIPT_DIR.parent
SRC_DIR = LLF_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 기본 체크포인트 경로를 지정하려면 아래 값을 변경하세요.
# None이면 --checkpoint 또는 --checkpoint-root 설정을 사용합니다.
DEFAULT_CHECKPOINT: Optional[Path] = LLF_DIR / "saves" / "qwen2_5vl-3b" / "rag_projector_v2" / "checkpoint-28000"

from llamafactory.chat.hf_engine import HuggingfaceEngine  # type: ignore  # noqa: E402
from llamafactory.hparams import get_infer_args  # type: ignore  # noqa: E402
from llamafactory.model.model_utils.rag_sequence import rag_sequence_generate  # type: ignore  # noqa: E402


@dataclass
class SpatialMQASample:
    """Minimal representation of a SpatialMQA entry."""

    image_path: str
    question: str
    options: list[str]
    answer: str
    prompt: str
    option_letters: list[str]


def find_latest_checkpoint(checkpoint_root: Path) -> tuple[Path, int]:
    """Return path to the numerically latest checkpoint directory."""

    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")

    pattern = re.compile(r"checkpoint-(\d+)")
    found: list[tuple[int, Path]] = []
    for entry in checkpoint_root.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.fullmatch(entry.name)
        if match:
            found.append((int(match.group(1)), entry))

    if not found:
        raise FileNotFoundError(f"No checkpoint-* directories under {checkpoint_root}")

    step, path = max(found, key=lambda item: item[0])
    return path, step


def resolve_image_path(image_root: Path, filename: str, search_order: tuple[str, ...]) -> Optional[Path]:
    """Locate the image file within the expected subdirectories."""

    for subdir in search_order:
        candidate = image_root / subdir / filename
        if candidate.exists():
            return candidate
    direct = image_root / filename
    return direct if direct.exists() else None


def build_prompt(question: str, options: list[str]) -> tuple[str, list[str]]:
    """Create a multiple-choice prompt that asks for a single-letter answer."""

    letters = [chr(ord("A") + idx) for idx in range(len(options))]
    lines = [f"{letter}. {option}" for letter, option in zip(letters, options)]
    prompt_lines = [
        question.strip(),
        "",
        "Options:",
        *lines,
        "",
        "Reply with only the single letter (e.g. A) that matches the correct option.",
    ]
    return "\n".join(prompt_lines), letters


def load_spatial_mqa_samples(
    dataset_path: Path,
    image_root: Path,
    limit: Optional[int],
    search_order: tuple[str, ...],
) -> list[SpatialMQASample]:
    """Load SpatialMQA entries and attach resolved image paths and prompts."""

    samples: list[SpatialMQASample] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            image_path = resolve_image_path(image_root, data["image"], search_order)
            if image_path is None:
                print(f"[WARN] Skipping sample; image not found: {data['image']}")
                continue
            prompt, option_letters = build_prompt(data["question"], data["options"])
            samples.append(
                SpatialMQASample(
                    image_path=str(image_path),
                    question=data["question"],
                    options=list(map(str, data["options"])),
                    answer=str(data["answer"]),
                    prompt=prompt,
                    option_letters=option_letters,
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise RuntimeError("No valid SpatialMQA samples were loaded.")
    return samples


def decode_generation(engine: HuggingfaceEngine, output_ids: torch.Tensor, input_length: int) -> str:
    """Decode model outputs excluding the prompt tokens."""

    generated_ids = output_ids[:, input_length:]
    return engine.tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()


def extract_prediction(text: str, sample: SpatialMQASample) -> tuple[Optional[str], Optional[str]]:
    """Derive the predicted option from the model's text response."""

    cleaned = text.strip()
    match = re.search(r"([A-Za-z])", cleaned)
    if match:
        letter = match.group(1).upper()
        if letter in sample.option_letters:
            idx = sample.option_letters.index(letter)
            return sample.options[idx], letter

    normalized = re.sub(r"[^a-z0-9 ]", "", cleaned.lower())
    for option in sample.options:
        if re.sub(r"[^a-z0-9 ]", "", option.lower()) in normalized:
            return option, None

    return None, None


def setup_engine(
    model_name_or_path: Union[str, Path],
    template: str,
    enable_rag: bool,
    rag_top_k: Optional[int],
    max_new_tokens: int,
    finetuning_type: str,
    device: Optional[str],
) -> HuggingfaceEngine:
    """Instantiate the HuggingfaceEngine with or without RAG."""

    infer_config: dict[str, Any] = {
        "model_name_or_path": str(model_name_or_path),
        "template": template,
        "infer_backend": "huggingface",
        "trust_remote_code": True,
        "enable_rag": enable_rag,
        "finetuning_type": finetuning_type,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 50,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "skip_special_tokens": True,
    }
    if enable_rag:
        if rag_top_k is not None:
            infer_config["rag_top_k"] = rag_top_k
        infer_config["rag_training_mode"] = "rag_only"

    model_args, data_args, finetuning_args, generating_args = get_infer_args(infer_config)
    generating_args.do_sample = False
    generating_args.temperature = 0.0
    generating_args.top_p = 1.0
    generating_args.max_new_tokens = max_new_tokens
    generating_args.repetition_penalty = 1.0
    generating_args.length_penalty = 1.0
    data_args.template = template
    finetuning_args.finetuning_type = finetuning_type

    if device is not None:
        model_args.device_map = {"": device}

    engine = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
    if enable_rag and rag_top_k is not None:
        engine.model.rag_top_k = rag_top_k
    return engine


def run_sample(
    engine: HuggingfaceEngine,
    sample: SpatialMQASample,
    rag_top_k: Optional[int],
    max_new_tokens: int,
    enable_rag: bool,
) -> tuple[str, float]:
    """Generate a response for one sample with optional RAG."""

    messages = [{"role": "user", "content": sample.prompt}]
    start = time.perf_counter()
    gen_kwargs, prompt_length = HuggingfaceEngine._process_args(
        engine.model,
        engine.tokenizer,
        engine.processor,
        engine.template,
        engine.generating_args,
        messages,
        images=[sample.image_path],
    )

    input_ids = gen_kwargs["inputs"]
    attention_mask = gen_kwargs.get("attention_mask")
    pixel_values = gen_kwargs.get("pixel_values")
    image_grid_thw = gen_kwargs.get("image_grid_thw")
    generation_config = gen_kwargs.get("generation_config")
    if generation_config is None:
        generation_config = engine.model.generation_config

    max_tokens = (generation_config.max_new_tokens if generation_config and generation_config.max_new_tokens is not None else max_new_tokens)
    do_sample = False
    temperature = 0.0
    top_p = 1.0
    pad_token_id = (generation_config.pad_token_id if generation_config and generation_config.pad_token_id is not None else engine.tokenizer.pad_token_id)
    eos_token_id = generation_config.eos_token_id if generation_config and generation_config.eos_token_id is not None else engine.tokenizer.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0]
    if isinstance(eos_token_id, torch.Tensor):
        eos_token_id = eos_token_id.item()

    if enable_rag:
        with torch.inference_mode():
            output_ids, _ = rag_sequence_generate(
                model=engine.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                rag_top_k=rag_top_k,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

        for cache_attr in ["_cached_vision_features", "_cached_layer_indices"]:
            if hasattr(engine.model, cache_attr):
                delattr(engine.model, cache_attr)

        selection_info = getattr(engine.model, "_last_rag_selected_indices", None)
        if selection_info:
            rel = selection_info.get("relative")
            abs_idx = selection_info.get("absolute")
            probs = selection_info.get("probabilities")
            print(
                "[RAG-SEQ DEBUG] selected_layers=",
                rel[0] if rel else rel,
                "absolute=",
                abs_idx[0] if abs_idx else abs_idx,
                "probs=",
                probs[0] if probs else probs,
            )
    else:
        hf_generate_kwargs: dict[str, Any] = {"inputs": input_ids, "generation_config": generation_config, "max_new_tokens": max_tokens}
        if attention_mask is not None:
            hf_generate_kwargs["attention_mask"] = attention_mask
        if pixel_values is not None:
            hf_generate_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            hf_generate_kwargs["image_grid_thw"] = image_grid_thw
        with torch.inference_mode():
            output_ids = engine.model.generate(**hf_generate_kwargs)

    text = decode_generation(engine, output_ids, input_ids.size(1) if input_ids.ndim == 2 else prompt_length)
    elapsed = time.perf_counter() - start
    return text, elapsed


def benchmark_model(
    model_identifier: str,
    engine: HuggingfaceEngine,
    samples: list[SpatialMQASample],
    max_new_tokens: int,
    rag_top_k: Optional[int],
    enable_rag: bool,
    dataset_file: Path,
    output_path: Path,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run SpatialMQA benchmark and persist results."""

    results: list[dict[str, Any]] = []
    correct = 0

    for idx, sample in enumerate(tqdm(samples, desc=f"SpatialMQA ({model_identifier})", unit="sample"), start=1):
        try:
            response, latency = run_sample(engine, sample, rag_top_k if enable_rag else None, max_new_tokens, enable_rag)
        except Exception as exc:  # pragma: no cover - inference-time failures
            print(f"[ERROR] Sample {idx} failed: {exc}")
            results.append(
                {
                    "index": idx - 1,
                    "image_path": sample.image_path,
                    "question": sample.question,
                    "options": sample.options,
                    "answer": sample.answer,
                    "response": None,
                    "predicted_option": None,
                    "predicted_letter": None,
                    "latency": None,
                    "error": str(exc),
                    "correct": False,
                }
            )
            continue

        predicted_option, predicted_letter = extract_prediction(response, sample)
        success = predicted_option == sample.answer if predicted_option is not None else False
        if success:
            correct += 1

        results.append(
            {
                "index": idx - 1,
                "image_path": sample.image_path,
                "question": sample.question,
                "options": sample.options,
                "answer": sample.answer,
                "response": response,
                "predicted_option": predicted_option,
                "predicted_letter": predicted_letter,
                "latency": latency,
                "correct": success,
            }
        )

    total = len(results)
    accuracy = correct / total if total else 0.0
    summary: dict[str, Any] = {
        "model_name_or_path": model_identifier,
        "rag_enabled": enable_rag,
        "dataset_file": str(dataset_file),
        "num_samples": total,
        "num_correct": correct,
        "accuracy": accuracy,
        "results": results,
    }
    if extra_metadata:
        summary.update(extra_metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[{model_identifier}] Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"[{model_identifier}] Detailed results saved to {output_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SpatialMQA inference with the latest checkpoint.")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=LLF_DIR / "saves" / "qwen2_5vl-3b" / "rag_projector_v2",
        help="Directory containing checkpoint-* folders",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Direct path to a specific checkpoint directory (overrides --checkpoint-root)",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=Path("/data3/DB/dataset/SpatialMQA/train.jsonl"),
        help="SpatialMQA JSONL file",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("/data3/DB/dataset/SpatialMQA/coco2017"),
        help="Root directory with COCO-style image folders",
    )
    parser.add_argument(
        "--image-subdirs",
        nargs="*",
        default=("test2017", "val2017", "train2017"),
        help="Priority order of sub-directories searched for images",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples (for quick tests)")
    parser.add_argument("--output", type=Path, default=Path("spatial_mqa_results.json"), help="Output JSON file")
    parser.add_argument("--template", type=str, default="qwen2_vl", help="Prompt template to use")
    parser.add_argument("--rag-top-k", type=int, default=3, help="Override rag_top_k during inference")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Maximum new tokens to generate")
    parser.add_argument(
        "--finetuning-type",
        type=str,
        default="full",
        choices=["full", "lora", "freeze"],
        help="Fine-tuning type used for the checkpoint (affects loader)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force all model weights onto a single device id (e.g. 'cuda:0' or 'cpu')",
    )

    parser.add_argument(
        "--evaluate-base",
        action="store_true",
        help="Also evaluate the base model without RAG.",
    )
    parser.add_argument(
        "--base-model-name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model name or path for baseline evaluation.",
    )
    parser.add_argument(
        "--base-output",
        type=Path,
        default=Path("spatial_mqa_base.json"),
        help="Output JSON file for baseline results.",
    )

    parser.add_argument(
        "--skip-tuned",
        action="store_true",
        help="Skip evaluation of the fine-tuned checkpoint.",
    )

    args = parser.parse_args()

    if args.skip_tuned and not args.evaluate_base:
        print("No evaluation requested: use --evaluate-base or remove --skip-tuned.")
        return

    if not args.skip_tuned and DEFAULT_CHECKPOINT is not None:
        args.checkpoint = DEFAULT_CHECKPOINT

    samples = load_spatial_mqa_samples(
        dataset_path=args.dataset_file,
        image_root=args.image_root,
        limit=args.limit,
        search_order=tuple(args.image_subdirs) if args.image_subdirs else tuple(),
    )
    print(f"Loaded {len(samples)} samples from {args.dataset_file}")

    tuned_engine: Optional[HuggingfaceEngine] = None
    checkpoint_path: Optional[Path] = None
    step: Optional[int] = None

    if not args.skip_tuned:
        if args.checkpoint is not None:
            checkpoint_path = args.checkpoint
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Specified checkpoint does not exist: {checkpoint_path}")
            step_match = re.search(r"checkpoint-(\d+)", checkpoint_path.name)
            step = int(step_match.group(1)) if step_match else None
        else:
            checkpoint_path, step = find_latest_checkpoint(args.checkpoint_root)

        step_display = step if step is not None else "unknown"
        print(f"Using checkpoint: {checkpoint_path} (step {step_display})")

        tuned_engine = setup_engine(
            model_name_or_path=checkpoint_path,
            template=args.template,
            enable_rag=True,
            rag_top_k=args.rag_top_k,
            max_new_tokens=args.max_new_tokens,
            finetuning_type=args.finetuning_type,
            device=args.device,
        )

        rag_top_k = args.rag_top_k if args.rag_top_k > 0 else getattr(tuned_engine.model, "rag_top_k", None)

        benchmark_model(
            model_identifier=str(checkpoint_path),
            engine=tuned_engine,
            samples=samples,
            max_new_tokens=args.max_new_tokens,
            rag_top_k=rag_top_k,
            enable_rag=True,
            dataset_file=args.dataset_file,
            output_path=args.output,
            extra_metadata={"checkpoint_step": step},
        )

    if args.evaluate_base:
        if tuned_engine is not None:
            del tuned_engine
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        base_engine = setup_engine(
            model_name_or_path=args.base_model_name,
            template=args.template,
            enable_rag=False,
            rag_top_k=None,
            max_new_tokens=args.max_new_tokens,
            finetuning_type="full",
            device=args.device,
        )

        benchmark_model(
            model_identifier=args.base_model_name,
            engine=base_engine,
            samples=samples,
            max_new_tokens=args.max_new_tokens,
            rag_top_k=None,
            enable_rag=False,
            dataset_file=args.dataset_file,
            output_path=args.base_output,
        )


if __name__ == "__main__":
    main()
