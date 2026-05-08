import json, torch, os
from PIL import Image
from tqdm import tqdm
from core.config import (
    TEST_JSONL, TEST_RESULTS_JSON, LORA_ADAPTER_DIR, 
    SYSTEM_PROMPT, ANCHOR_REAL_PATH, ANCHOR_FAKE_PATH, GENERATION_KWARGS
)
from core.model_loader import get_processor, load_model_for_inference

def extract_label(text): 
    return 0 if "pristine" in text.lower() or "no," in text.lower() else 1

if __name__ == "__main__":
    processor = get_processor()
    model = load_model_for_inference(LORA_ADAPTER_DIR)
    
    img_real = Image.open(ANCHOR_REAL_PATH).convert("RGB").resize((448, 448))
    img_fake = Image.open(ANCHOR_FAKE_PATH).convert("RGB").resize((448, 448))
    
    prepared_test = TEST_JSONL.replace(".jsonl", "_prepared.jsonl")
    with open(prepared_test, 'r', encoding='utf-8') as f: 
        lines = f.readlines()
        
    results = []

    for line in tqdm(lines, desc="Inference"):
        data = json.loads(line.strip())
        img_path = data['messages'][0]['content'][0]['image']
        prompt_text = data['messages'][0]['content'][1]['text']
        true_text = data['messages'][1]['content'][0]['text']
        
        try:
            target_img = Image.open(img_path).convert("RGB")
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "image", "image": img_real}, {"type": "text", "text": prompt_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": "No, this is a pristine image. The features are natural."}]},
                {"role": "user", "content": [{"type": "image", "image": img_fake}, {"type": "text", "text": prompt_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": "There is an unusual trembling of the eyebrows and eyes. The nose is slightly deformed."}]},
                {"role": "user", "content": [{"type": "image", "image": target_img}, {"type": "text", "text": "Analyze this target image: " + prompt_text}]}
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text], images=[img_real, img_fake, target_img], 
                padding=True, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                out_ids = model.generate(**inputs, **GENERATION_KWARGS)
            
            pred_text = processor.batch_decode([o[len(i):] for i, o in zip(inputs.input_ids, out_ids)], skip_special_tokens=True)[0]
            
            results.append({
                "image_path": img_path, 
                "ground_truth": {"label": extract_label(true_text), "text": true_text},
                "prediction": {"label": extract_label(pred_text), "text": pred_text}, 
                "is_correct_class": extract_label(true_text) == extract_label(pred_text)
            })
        except Exception as e:
            print(f"Pass {img_path}: {e}")
            continue

    with open(TEST_RESULTS_JSON, 'w', encoding='utf-8') as f: 
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to: {TEST_RESULTS_JSON}")