import os
import sys
import time
import glob
from typing import List

import torch


# Add LLaMA-Factory src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLF_SRC = os.path.join(BASE_DIR, "LLaMA-Factory", "src")
if LLF_SRC not in sys.path:
    sys.path.insert(0, LLF_SRC)

# LLaMA-Factory imports
from llamafactory.hparams import get_infer_args  # type: ignore  # noqa: E402
from llamafactory.chat.hf_engine import HuggingfaceEngine  # type: ignore  # noqa: E402
from llamafactory.model.model_utils.rag_sequence import rag_sequence_generate  # type: ignore  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

# Using rag_sequence_generate from llamafactory instead


def run_single_image(engine: HuggingfaceEngine, image_path: str, prompt: str, rag_top_k: int | None = None,
                     max_new_tokens: int | None = None) -> tuple[str, int, float]:
    """Prepare MM inputs and generate with rag_sequence_generate.

    Returns: (text, generated_tokens, elapsed_seconds)
    """
    messages = [{"role": "user", "content": prompt}]
    images: List[str] = [image_path]

    gen_kwargs, prompt_length = HuggingfaceEngine._process_args(  # type: ignore[attr-defined]
        engine.model,
        engine.tokenizer,
        engine.processor,
        engine.template,
        engine.generating_args,
        messages,
        system=None,
        tools=None,
        images=images,
        videos=None,
        audios=None,
        input_kwargs={},
    )

    input_ids = gen_kwargs.pop("inputs")
    attention_mask = gen_kwargs.pop("attention_mask", None)
    pixel_values = gen_kwargs.pop("pixel_values", None)
    image_grid_thw = gen_kwargs.pop("image_grid_thw", None)

    # Decide max_new_tokens
    max_new = max_new_tokens or gen_kwargs.get("generation_config", engine.model.generation_config).max_new_tokens
    if max_new is None:
        max_new = 256

    # Move tensors
    device = engine.model.device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values.to(device, dtype=engine.model.dtype)
    if isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = image_grid_thw.to(device)

    start = time.time()
    with torch.inference_mode():
        # Check if RAG is enabled
        print(f"[DEBUG] RAG enabled: {getattr(engine.model, 'enable_rag', False)}")
        print(f"[DEBUG] RAG top_k: {getattr(engine.model, 'rag_top_k', None)}")
        
        # Check if RAG modules are actually being used
        print(f"[DEBUG] Checking RAG module usage...")
        
        # Try to access RAG modules directly
        if hasattr(engine.model, 'rag_modules'):
            print(f"[DEBUG] RAG modules found: {type(engine.model.rag_modules)}")
            if hasattr(engine.model.rag_modules, 'score_adapter'):
                print(f"[DEBUG] Score adapter found: {engine.model.rag_modules.score_adapter}")
        else:
            print(f"[DEBUG] No rag_modules attribute found")
            
        # Check if rag_projection exists
        if hasattr(engine.model, 'rag_projection'):
            print(f"[DEBUG] rag_projection found: {type(engine.model.rag_projection)}")
        else:
            print(f"[DEBUG] No rag_projection attribute found")
            
        # Use our custom RAG generate function
        print(f"[DEBUG] Using custom RAG generate function")
        
        # Handle generation_config properly
        if hasattr(engine.generating_args, '__dict__'):
            gen_config = engine.generating_args.__dict__
        else:
            gen_config = engine.generating_args
            
        output_ids = rag_generate(
            model=engine.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            rag_top_k=rag_top_k,
            max_new_tokens=int(max_new),
            generation_config=gen_config,
            logits_processor=None,
            stopping_criteria=None,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        print(f"[DEBUG] RAG generation completed")
    elapsed = time.time() - start

    # Debug: Check dimensions
    print(f"[DEBUG] output_ids shape: {output_ids.shape}")
    print(f"[DEBUG] prompt_length: {prompt_length}")
    print(f"[DEBUG] input_ids shape: {input_ids.shape}")
    
    # Calculate actual new tokens
    actual_new_tokens = output_ids.size(1) - input_ids.size(1)
    print(f"[DEBUG] Actual new tokens: {actual_new_tokens}")
    
    # Extract only the new tokens
    if actual_new_tokens > 0:
        response_ids = output_ids[:, input_ids.size(1):]
        text = engine.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
    else:
        # Fallback: use the original method
        response_ids = output_ids[:, prompt_length:]
        text = engine.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
    
    print(f"[DEBUG] Response text: '{text}'")
    return text, actual_new_tokens, elapsed


def main() -> None:
    # Configs (edit here as you like)
    infer_yaml = os.path.join(BASE_DIR, "LLaMA-Factory", "examples", "inference", "qwen2_5vl_rag_sequence_final.yaml")
    images_dir = os.path.join(BASE_DIR, "test_images")
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    # Use an English prompt for better results
    prompt = "What do you see in this image? Please describe it briefly."
    rag_top_k = 3  # Enable RAG
    max_new_tokens = 256  # Increase max tokens for better quality

    # Load model via YAML (parse YAML to dict for get_infer_args)
    cfg = OmegaConf.load(infer_yaml)
    cfg_dict = OmegaConf.to_container(cfg)
    model_args, data_args, finetuning_args, generating_args = get_infer_args(cfg_dict)
    # Force single-device placement to avoid cross-device tensor ops during RAG hooks
    if torch.cuda.is_available():
        # respect CUDA_VISIBLE_DEVICES; map to first visible GPU index (override with GPU env)
        try:
            gpu_idx = int(os.getenv("GPU", "0"))
        except ValueError:
            gpu_idx = 0
        model_args.device_map = {"": gpu_idx}
    else:
        model_args.device_map = {"": torch.device("cpu")}
    # Build engine (loads tokenizer/model internally using our device_map)
    engine = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)

    # Collect images
    image_paths: list[str] = []
    for pat in patterns:
        image_paths.extend(glob.glob(os.path.join(images_dir, pat)))
    image_paths = sorted(image_paths)
    if not image_paths:
        print(f"No images found in {images_dir} with patterns {patterns}")
        return

    print(f"Model: {model_args.model_name_or_path}")
    print(
        f"RAG enabled={getattr(engine.model, 'enable_rag', False)} rag_top_k={getattr(engine.model, 'rag_top_k', None)}"
    )
    print(f"Images: {len(image_paths)} from {images_dir}\n")

    for p in image_paths:
        print(f"=== {os.path.basename(p)} ===")
        try:
            text, gen_toks, secs = run_single_image(
                engine, p, prompt, rag_top_k=rag_top_k, max_new_tokens=max_new_tokens
            )
            tps = gen_toks / secs if secs > 0 else float("inf")
            print(text.strip())
            print(f"[generated_tokens={gen_toks} time={secs:.2f}s tok/s={tps:.1f}]\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
