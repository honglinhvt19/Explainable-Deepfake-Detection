import gradio as gr
import torch
import cv2
from PIL import Image

from core.config import LORA_ADAPTER_DIR, SYSTEM_PROMPT, GENERATION_KWARGS
from core.model_loader import get_processor, load_model_for_inference

print("Initializing...")
processor = get_processor()
model = load_model_for_inference(LORA_ADAPTER_DIR)

def extract_frames(video_path, num_frames=128):
    cap = cv2.VideoCapture(video_path)
    frames, interval = [], max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // num_frames)
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i * interval, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1))
        ret, frame = cap.read()
        if ret: 
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((448, 448)))
    cap.release()
    return frames

def analyze(input_image, input_video, prompt_text):
    if not input_image and not input_video: 
        return "Please upload an image or video."
    try:
        media_inputs = [input_image.convert("RGB")] if input_image else extract_frames(input_video)
        content = [{"type": "image", "image": img} for img in media_inputs] + [{"type": "text", "text": prompt_text}]
        
        text = processor.apply_chat_template([
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content}
        ], tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text], images=media_inputs, padding=True, return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            out_ids = model.generate(**inputs, **GENERATION_KWARGS)
            
        res = processor.batch_decode([o[len(i):] for i, o in zip(inputs.input_ids, out_ids)], skip_special_tokens=True)[0]
        status = "🟢 REAL" if "pristine" in res.lower() or "no," in res.lower() else "🔴 DEEPFAKE"
        
        return f"### {status}\n**Details (XAI):**\n{res}"
    except Exception as e: 
        return f"⚠️ Error: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft(), title="Deepfake XAI") as app:
    gr.Markdown("# Deepfake Detection System (XAI)*")
    with gr.Row():
        with gr.Column():
            in_img = gr.Image(type="pil", label="Image")
            in_vid = gr.Video(label="Video")
            in_prompt = gr.Textbox(label="Prompt", value="Carefully analyze this media. Is this a deepfake? If yes, provide a detailed explanation of the visual artifacts.")
            btn = gr.Button("🔍 Analyze Now", variant="primary")
        out = gr.Markdown("*Waiting for data...*")
    
    btn.click(analyze, [in_img, in_vid, in_prompt], out)

if __name__ == "__main__": 
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)