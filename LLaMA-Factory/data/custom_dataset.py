import json
import os

def convert_spatial_mqa_to_llamafactory():
    # 원본 데이터 읽기
    with open('/data3/DB/dataset/SpatialMQA/train.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    # LLaMA-Factory ShareGPT 형식으로 변환
    converted_data = []
    for item in data:
        converted_item = {
            'messages': [
                {
                    'content': f'<image>{item["question"]}',
                    'role': 'user'
                },
                {
                    'content': item["answer"],
                    'role': 'assistant'
                }
            ],
            'images': [f'coco2017/test2017/{item["image"]}']
        }
        converted_data.append(converted_item)
    
    
    # 변환된 데이터 저장
    with open('/data3/DB/dataset/SpatialMQA/spatial_mqa_llamafactory.json', 'w') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f'변환 완료: {len(converted_data)}개 데이터')
    print('샘플 데이터:')
    print(json.dumps(converted_data[0], indent=2, ensure_ascii=False))


def main():
    convert_spatial_mqa_to_llamafactory()

if __name__ == "__main__":
    print("변환 시작")
    main()


    