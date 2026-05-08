import os
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from core.config import TRAIN_JSONL, VAL_JSONL, TRAIN_OUTPUT_DIR, TRAIN_ARGS_DICT
from core.model_loader import get_processor, load_model_for_training

def mask_instruction_tokens(input_ids, processor):
    labels = input_ids.clone()
    header_ids = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    header_tensor = torch.tensor(header_ids)
    mask_end = 0
    for i in range(input_ids.shape[0] - len(header_ids), -1, -1):
        if torch.equal(input_ids[i : i + len(header_ids)], header_tensor):
            mask_end = i + len(header_ids)
            break
    labels[:mask_end] = -100
    return labels

def multimodal_data_collator(examples):
    input_ids_list, attention_mask_list, labels_list, pixel_values_list, image_grid_thw_list = [], [], [], [], []
    for example in examples:
        messages = example["messages"]
        sample_images = [
            Image.open(c["image"]).convert("RGB") 
            for msg in messages if isinstance(msg.get("content"), list)
            for c in msg["content"] if c.get("type") == "image"
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        try:
            inputs = processor(
                text=[text], images=sample_images if sample_images else None,
                return_tensors="pt", padding=False, min_pixels=256*28*28, max_pixels=512*28*28
            )
            labels = mask_instruction_tokens(inputs["input_ids"][0], processor)
            input_ids_list.append(inputs["input_ids"][0])
            attention_mask_list.append(inputs["attention_mask"][0])
            labels_list.append(labels)
            if "pixel_values" in inputs:
                pixel_values_list.append(inputs["pixel_values"])
                image_grid_thw_list.append(inputs["image_grid_thw"])
        except Exception: 
            continue

    batch = {
        "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=processor.tokenizer.pad_token_id),
        "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
        "labels": pad_sequence(labels_list, batch_first=True, padding_value=-100),
    }
    if pixel_values_list:
        batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        batch["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
    return batch

if __name__ == "__main__":
    print("Loading processor and model for training...")
    processor = get_processor()
    model = load_model_for_training()
    
    prepared_train = TRAIN_JSONL.replace(".jsonl", "_prepared.jsonl")
    prepared_val = VAL_JSONL.replace(".jsonl", "_prepared.jsonl")
    dataset = load_dataset("json", data_files={"train": prepared_train, "val": prepared_val})

    training_args = TrainingArguments(
        output_dir=TRAIN_OUTPUT_DIR,
        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss", 
        greater_is_better=False,
        logging_steps=10, eval_strategy="steps", eval_steps=200, 
        save_strategy="steps", save_steps=400, save_total_limit=3,
        gradient_checkpointing=True, report_to="none", 
        predict_with_generate=False, generation_max_length=128, 
        remove_unused_columns=False,
        **TRAIN_ARGS_DICT
    )

    trainer = Trainer(
        model=model, args=training_args, 
        train_dataset=dataset["train"], eval_dataset=dataset["val"],
        data_collator=multimodal_data_collator, 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("Starting training...")
    trainer.train()
    
    final_model_path = os.path.join(TRAIN_OUTPUT_DIR, "final_lora_model")
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    print(f"Completed! Model has been saved at {final_model_path}")