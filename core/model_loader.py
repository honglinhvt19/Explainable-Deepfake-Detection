import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from core.config import BASE_MODEL_ID, LORA_CONFIG_DICT

def get_processor():
    return AutoProcessor.from_pretrained(BASE_MODEL_ID)

def load_model_for_training():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    lora_config = LoraConfig(**LORA_CONFIG_DICT)
    model = get_peft_model(model, lora_config)
    return model

def load_model_for_inference(adapter_path: str):
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, 
        torch_dtype=torch.float16, 
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    return model