#!/usr/bin/env python3
"""
RAG-Sequence 모델 Inference 및 벤치마크 스크립트
"""

import torch
import json
import time
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoProcessor
from llamafactory.model import load_model_and_tokenizer
from llamafactory.model.model_utils.rag_sequence import rag_sequence_generate
import argparse
import os

def load_rag_model(model_path: str, device: str = "cuda"):
    """RAG-Sequence 모델 로드"""
    print(f"Loading model from: {model_path}")
    
    # 모델과 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_path,
        template="qwen2_vl",
        finetuning_type="lora",
        device_map=device,
        torch_dtype=torch.float16,
    )
    
    # RAG 모듈들 로드
    from llamafactory.model.model_utils.rag_sequence import attach_rag_modules
    attach_rag_modules(model)
    
    # 체크포인트 로드
    checkpoint_path = os.path.join(model_path, "checkpoint-3000")  # 최신 체크포인트
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location=device))
    
    model.eval()
    return model, tokenizer

def inference_single_sample(model, tokenizer, image_path: str, question: str, max_new_tokens: int = 512):
    """단일 샘플 inference"""
    
    # 이미지 로드
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    
    # 텍스트 토크나이징
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]}
    ]
    
    # 토크나이징
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    
    # 이미지 전처리
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    pixel_values = processor(images=[image], return_tensors="pt")["pixel_values"]
    
    # RAG-Sequence 생성
    with torch.no_grad():
        outputs = rag_sequence_generate(
            model=model,
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            pixel_values=pixel_values.to(model.device),
            image_grid_thw=None,  # 자동 감지
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # 응답 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def benchmark_model(model, tokenizer, test_data: List[Dict], output_file: str = "benchmark_results.json"):
    """모델 벤치마크 실행"""
    
    results = []
    total_time = 0
    
    print(f"Running benchmark on {len(test_data)} samples...")
    
    for i, sample in enumerate(test_data):
        print(f"Processing sample {i+1}/{len(test_data)}")
        
        start_time = time.time()
        
        try:
            response = inference_single_sample(
                model=model,
                tokenizer=tokenizer,
                image_path=sample["image_path"],
                question=sample["question"],
                max_new_tokens=sample.get("max_tokens", 512)
            )
            
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            
            result = {
                "sample_id": i,
                "image_path": sample["image_path"],
                "question": sample["question"],
                "response": response,
                "inference_time": inference_time,
                "success": True
            }
            
        except Exception as e:
            result = {
                "sample_id": i,
                "image_path": sample["image_path"],
                "question": sample["question"],
                "response": None,
                "error": str(e),
                "success": False
            }
        
        results.append(result)
    
    # 결과 저장
    benchmark_summary = {
        "total_samples": len(test_data),
        "successful_samples": sum(1 for r in results if r["success"]),
        "total_time": total_time,
        "average_time_per_sample": total_time / len(test_data),
        "results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, ensure_ascii=False, indent=2)
    
    print(f"Benchmark completed!")
    print(f"Total samples: {len(test_data)}")
    print(f"Successful: {benchmark_summary['successful_samples']}")
    print(f"Average time per sample: {benchmark_summary['average_time_per_sample']:.2f}s")
    print(f"Results saved to: {output_file}")
    
    return benchmark_summary

def main():
    parser = argparse.ArgumentParser(description="RAG-Sequence Model Inference and Benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data JSON file")
    parser.add_argument("--output_file", type=str, default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 테스트 데이터 로드
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    # 모델 로드
    model, tokenizer = load_rag_model(args.model_path, args.device)
    
    # 벤치마크 실행
    benchmark_model(model, tokenizer, test_data, args.output_file)

if __name__ == "__main__":
    main()

