import os
import torch
from dotenv import load_dotenv

load_dotenv()

def get_env_var(var_name, default=None):
    val = os.getenv(var_name, default)
    if val is None:
        print(f"[WARNING] Environment variable '{var_name}' is not set. Using default: {default}")
    return val

ROOT_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# --- Training Paths ---
TRAIN_JSONL = get_env_var("TRAIN_JSONL", "./data/train.jsonl")
VAL_JSONL = get_env_var("VAL_JSONL", "./data/val.jsonl")
TRAIN_OUTPUT_DIR = os.path.join(ROOT_PROJECT_DIR, "output_lora") # Tự động tạo thư mục output_lora

# --- Test & Inference Paths ---
TEST_JSONL = get_env_var("TEST_JSONL", "./data/test.jsonl")
TEST_RESULTS_JSON = os.path.join(ROOT_PROJECT_DIR, "test_results.json")
LORA_ADAPTER_DIR = get_env_var("LORA_DIR", "Hong-Linh/qwen3-vl-xai-deepfake")

# ==========================================
# 2. PROMPTS & FEW-SHOT ANCHORS
# ==========================================
SYSTEM_PROMPT = """Analyze the visual content as a highly cautious forensic expert. 
WARNING: DO NOT confuse natural camera noise, low resolution, motion blur, or normal lighting shadows with deepfake artifacts. 
Only declare it a deepfake if you see clear, undeniable manipulation (e.g., distorted facial features, blending errors). 
If the content looks natural, you MUST output: 'No, this is a pristine media. The features are natural.'"""

ANCHOR_REAL_PATH = os.path.join(ROOT_PROJECT_DIR, "demo_images", "sample_real.jpg")
ANCHOR_FAKE_PATH = os.path.join(ROOT_PROJECT_DIR, "demo_images", "sample_fake.jpg")

# ==========================================
# 3. TRAINING ARGS
# ==========================================
LORA_CONFIG_DICT = {
    "r": 64,
    "lora_alpha": 128,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAIN_ARGS_DICT = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "num_train_epochs": 5,
    "bf16": True,
}

# ==========================================
# 4. GENERATION ARGS
# ==========================================
GENERATION_KWARGS = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.08
}