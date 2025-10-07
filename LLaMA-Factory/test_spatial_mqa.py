#!/usr/bin/env python3

import os
import json
from PIL import Image

def test_spatial_mqa_data():
    """SpatialMQA 데이터 확인 및 간단한 테스트"""
    
    data_file = "/data3/DB/dataset/SpatialMQA/train.jsonl"
    image_base_path = "/data3/DB/dataset/SpatialMQA/coco2017/test2017"
    
    print("=== SpatialMQA 데이터 테스트 ===")
    
    # 처음 5개 데이터 확인
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            
            data = json.loads(line.strip())
            image_filename = data['image']
            question = data['question']
            options = data['options']
            answer = data['answer']
            
            image_path = os.path.join(image_base_path, image_filename)
            image_exists = os.path.exists(image_path)
            
            print(f"\n--- Sample {i+1} ---")
            print(f"Image: {image_filename} {'✅' if image_exists else '❌'}")
            print(f"Question: {question}")
            print(f"Options: {options}")
            print(f"Answer: {answer}")
            
            if image_exists:
                try:
                    img = Image.open(image_path)
                    print(f"Image size: {img.size}")
                except Exception as e:
                    print(f"Image load error: {e}")
    
    print("\n=== 평가 시스템 테스트 ===")
    print("이제 다음 명령어로 RAG 모델을 평가할 수 있습니다:")
    print("llamafactory-cli eval examples/eval_rag.yaml")

if __name__ == "__main__":
    test_spatial_mqa_data()
