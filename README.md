# Explainable Deepfake Detection using Large Vision-Language Models (LVLMs) 🔍🤖

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Introduction
Traditional deepfake detection systems operate as "black boxes," providing a simple Real/Fake binary output without justification. This project addresses this limitation by fine-tuning a Large Vision-Language Model (**Qwen3-VL-8B-Instruct**) using Low-Rank Adaptation (LoRA) to act as a forensic expert. 

**Core Philosophy:** The model focuses strictly on providing detailed **semantic explanations (XAI)** of visual artifacts (e.g., unnatural blending, abnormal lighting, skin texture inconsistencies). Crucially, this methodology is designed to process holistic spatial features and **does not involve the extraction of visual coordinates or bounding boxes**.

## ✨ Key Features
* **Semantic Explainability:** Outputs human-readable forensic analyses of deepfake media.
* **Few-Shot Contrastive Prompting:** Utilizes anchor images (pristine vs. fake) to enhance the model's analytical reasoning.
* **Comprehensive XAI Evaluation:** Incorporates an advanced LLM-as-a-Judge metric (`Prometheus-2-7B`) alongside standard NLP metrics (BERTScore, ROUGE-L, BLEU) to quantify explanation quality.
* **Interactive UI:** Built-in Gradio web application for real-time video and image inference.
* **Hardware Optimized:** Configured with Flash Attention 2, bfloat16 precision, and Gradient Checkpointing to run efficiently on single-GPU setups (e.g., NVIDIA RTX 4090).

## 🗂️ Project Structure
\`\`\`text
Explainable-Deepfake-Detection/
├── core/
│   ├── config.py             # Global constants and hyperparameter configs
│   ├── model_loader.py       # Base model and LoRA adapter loading logic
│   ├── extract_frame.py      # Face cropping using MTCNN
│   └── data_prepare.py    
├── demo_images/              # Anchor samples for few-shot prompting
├── train.py                  # LoRA fine-tuning script
├── test.py                   # Inference script to generate predictions
├── eval_metrics.py           # Classification and XAI metrics computation
├── web_app.py                # Gradio-based user interface
├── requirements.txt          # Dependencies
└── .env                      # Environment variables
\`\`\`

## ⚙️ Installation

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/your-username/Explainable-Deepfake-Detection.git
   cd Explainable-Deepfake-Detection
   \`\`\`

2. **Install dependencies:**
   It is recommended to use a virtual environment (`conda` or `venv`).
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Environment Setup:**
   Create a `.env` file in the root directory and configure your paths:
   \`\`\`env
   TRAIN_JSONL=./data/train.jsonl
   VAL_JSONL=./data/val.jsonl
   TEST_JSONL=./data/test.jsonl
   LORA_DIR=Hong-Linh/qwen3-vl-xai-deepfake
   TRAIN_IMAGE_DIR=/path/to/your/train/images
   TEST_IMAGE_DIR=/path/to/your/test/images
   \`\`\`

## 🚀 Usage

### 1. Data Preprocessing
Extract aligned face frames from your raw video dataset:
\`\`\`bash
python scripts/extract_frame.py --video_dir /path/to/videos --output_dir /path/to/frames --frames_per_video 5
\`\`\`
Update your `jsonl` files with the correct absolute image paths:
\`\`\`bash
python scripts/data_prepare.py
\`\`\`

### 2. Training (LoRA Fine-tuning)
Start the fine-tuning process. The model will save checkpoints in the `output_lora` directory.
\`\`\`bash
python train.py
\`\`\`

### 3. Inference & Testing
Run the model on your test set to generate predictions and explanations:
\`\`\`bash
python test.py
\`\`\`
*(Results will be saved to `test_results.json`)*

### 4. Evaluation
Calculate both Classification Metrics (Accuracy, F1) and XAI Metrics (Prometheus-2, BERTScore, ROUGE-L):
\`\`\`bash
python eval_metrics.py
\`\`\`
*(Generates `detailed_evaluation.csv` and `evaluation_report.md`)*

### 5. Web UI Demo
Launch the interactive Gradio interface:
\`\`\`bash
python web_app.py
\`\`\`
Access the UI via your browser at `http://localhost:7860`.

## 📊 Evaluation Metrics
The system is evaluated on two primary fronts:
1. **Detection Accuracy:** Binary classification performance (Real vs. Fake).
2. **Explanation Quality:** Assessed using **Prometheus-7b-v2.0** (LLM-as-a-Judge) strictly scoring against predefined rubrics to measure hallucination rates and alignment with human expert ground truth.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.