import json
import os
from core.config import get_env_var, TRAIN_JSONL, TEST_JSONL

TRAIN_IMG_DIR = get_env_var("TRAIN_IMAGE_DIR", "/root/dataset")
TEST_IMG_DIR = get_env_var("TEST_IMAGE_DIR", "/workspace/dataset_test")

TRAIN_OUT = TRAIN_JSONL.replace(".jsonl", "_prepared.jsonl")
TEST_OUT = TEST_JSONL.replace(".jsonl", "_prepared.jsonl")

def update_paths(input_file, output_file, image_dir):
    if not os.path.exists(input_file):
        print(f"[ERROR] File not found: {input_file}")
        return
        
    success_count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip(): continue
            data = json.loads(line.strip())
            for msg in data.get('messages', []):
                if msg.get('role') == 'user':
                    for content in msg.get('content', []):
                        if content.get('type') == 'image' and 'image' in content:
                            filename = os.path.basename(content['image'])
                            content['image'] = os.path.join(image_dir, filename)
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
            success_count += 1
    print(f"Prepared {success_count} samples -> {output_file}")

if __name__ == "__main__":
    print("="*50)
    print("Preparing data by updating image paths...")
    update_paths(TRAIN_JSONL, TRAIN_OUT, TRAIN_IMG_DIR)
    update_paths(TEST_JSONL, TEST_OUT, TEST_IMG_DIR)
    print("="*50)