#!/usr/bin/env python3
"""
RAG 모델 간단 테스트 스크립트
테스트용 이미지를 다운로드하고 RAG 모델을 테스트합니다.
"""

import os
import sys
import requests
from PIL import Image
import subprocess

def download_test_image(url: str, save_path: str):
    """테스트 이미지를 다운로드합니다."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded: {save_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def create_test_images():
    """테스트용 이미지들을 준비합니다."""
    
    # 테스트 이미지 디렉토리 생성
    test_dir = "/data3/jykim/Projects/VLM/test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # 테스트 이미지 URL들 (공개 이미지)
    test_images = {
        "cat.jpg": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=512&h=512&fit=crop",
        "dog.jpg": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=512&h=512&fit=crop", 
        "scene.jpg": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=512&h=512&fit=crop",
        "object.jpg": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=512&h=512&fit=crop",
        "food.jpg": "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=512&h=512&fit=crop"
    }
    
    print("📥 Downloading test images...")
    
    downloaded_count = 0
    for filename, url in test_images.items():
        save_path = os.path.join(test_dir, filename)
        
        if not os.path.exists(save_path):
            if download_test_image(url, save_path):
                downloaded_count += 1
        else:
            print(f"⏭️  Already exists: {save_path}")
            downloaded_count += 1
    
    print(f"✅ {downloaded_count}/{len(test_images)} images ready!")
    return test_dir

def run_simple_test():
    """간단한 테스트를 실행합니다."""
    
    # 테스트 이미지 준비
    test_dir = create_test_images()
    
    # 사용 가능한 모델들
    models = {
        "1": "/data3/jykim/Projects/VLM/LLaMA-Factory/saves/qwen2_5vl-3b/rag_sequence_top3/checkpoint-6000",
        "2": "/data3/jykim/Projects/VLM/LLaMA-Factory/saves/qwen2_5vl-3b/rag_sequence/sft",
        "3": "/data3/jykim/Projects/VLM/LLaMA-Factory/saves/qwen2_5vl-3b/rag_sequence_top3/checkpoint-4000"
    }
    
    print("\n🤖 RAG Model Simple Test")
    print("=" * 30)
    print("Available models:")
    for key, path in models.items():
        print(f"  {key}. {path}")
    
    choice = input("\nSelect model (1-3): ").strip()
    
    if choice not in models:
        print("❌ Invalid choice!")
        return
    
    model_path = models[choice]
    
    # 간단한 테스트 케이스
    test_cases = [
        {
            "image": os.path.join(test_dir, "cat.jpg"),
            "prompt": "이 이미지에서 무엇을 볼 수 있나요?",
            "description": "고양이 이미지 테스트"
        },
        {
            "image": os.path.join(test_dir, "dog.jpg"), 
            "prompt": "이 동물의 색깔은 무엇인가요?",
            "description": "개 이미지 테스트"
        }
    ]
    
    print(f"\n🧪 Running {len(test_cases)} test cases...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['description']}")
        print(f"📸 Image: {test_case['image']}")
        print(f"❓ Prompt: {test_case['prompt']}")
        print("-" * 50)
        
        # RAG inference 실행
        cmd = [
            "python", "rag_inference.py",
            "--model_path", model_path,
            "--image_path", test_case["image"],
            "--prompt", test_case["prompt"],
            "--temperature", "0.7",
            "--max_new_tokens", "256"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("✅ Test completed!")
                # 출력에서 응답 부분만 추출
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if "🤖 Response:" in line:
                        print(line)
                        break
                
                # 응답을 찾지 못한 경우 전체 출력 표시
                if not response_found:
                    print("📝 Full output:")
                    print(result.stdout)
            else:
                print(f"❌ Test failed!")
                print(f"Error: {result.stderr}")
                print(f"Output: {result.stdout}")
                
        except subprocess.TimeoutExpired:
            print("⏰ Test timed out!")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """메인 함수"""
    print("🚀 RAG Model Simple Test Setup")
    print("=" * 40)
    
    try:
        run_simple_test()
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
