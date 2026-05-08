import json, re, torch, os, nltk
import pandas as pd
import logging
import transformers
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.config import TEST_RESULTS_JSON, ROOT_PROJECT_DIR

transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
nltk.download('punkt', quiet=True)

OUTPUT_CSV = os.path.join(ROOT_PROJECT_DIR, "detailed_evaluation.csv")
OUTPUT_MD = os.path.join(ROOT_PROJECT_DIR, "evaluation_report.md")
PROMETHEUS_MODEL_ID = "prometheus-eval/prometheus-7b-v2.0"

if __name__ == "__main__":
    if not os.path.exists(TEST_RESULTS_JSON):
        print(f"Test results file not found: {TEST_RESULTS_JSON}. Please run test.py first to generate the results.")
        exit()

    with open(TEST_RESULTS_JSON, 'r', encoding='utf-8') as f:
        results = json.load(f)

    y_true = [item['ground_truth']['label'] for item in results]
    y_pred = [item['prediction']['label'] for item in results]

    fake_indices = [i for i, label in enumerate(y_true) if label == 1]
    fake_truths = [results[i]['ground_truth']['text'] for i in fake_indices]
    fake_preds  = [results[i]['prediction']['text'] for i in fake_indices]

    # --- 1. CLASSIFICATION METRICS ---
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # --- 2. XAI METRICS ---
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method1
    rougeL_scores, bleu_scores = [], []
    for ref, pred in zip(fake_truths, fake_preds):
        rougeL_scores.append(scorer.score(ref, pred)['rougeL'].fmeasure)
        bleu_scores.append(sentence_bleu([nltk.word_tokenize(ref.lower())], nltk.word_tokenize(pred.lower()), smoothing_function=smoothie))
    
    avg_rougeL = sum(rougeL_scores) / max(len(rougeL_scores), 1)
    avg_bleu = sum(bleu_scores) / max(len(bleu_scores), 1)
    
    print("Calculating BERTScore...")
    _, _, F1_bert = bert_score(fake_preds, fake_truths, lang="en", verbose=False) if fake_preds else (0,0,torch.tensor(0.0))
    avg_bert_f1 = F1_bert.mean().item()

    # --- 3. PROMETHEUS-2 EVALUATION ---
    num_samples = min(100, len(fake_preds))
    avg_prom_score = 0.0
    if num_samples > 0:
        print("Initializing Prometheus-2...")
        tokenizer = AutoTokenizer.from_pretrained(PROMETHEUS_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(PROMETHEUS_MODEL_ID, torch_dtype=torch.float16, device_map="auto")
        
        total_prom, successful_evals = 0, 0
        for i in tqdm(range(num_samples), desc="Prometheus Grading"):
            prompt = f"###Task Description:\nAn instruction, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.\n1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric.\n2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.\n3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"\n\n###The instruction to evaluate:\nAnalyze this image. Is it a deepfake? If yes, provide a detailed explanation of the visual artifacts.\n\n###Response to evaluate:\n{fake_preds[i]}\n\n###Reference Answer (Score 5):\n{fake_truths[i]}\n\n###Score Rubrics:\n[1] The response completely hallucinates visual artifacts.\n[2] The response mentions many incorrect visual artifacts.\n[3] The response identifies some correct artifacts but includes minor hallucinations.\n[4] The response is mostly accurate and aligns with the reference answer.\n[5] The response perfectly aligns with the reference answer with zero hallucination.\n\n###Feedback: "
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, temperature=0.0, do_sample=False)
            resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            match = re.search(r'\[RESULT\]\s*(\d)', resp)
            if match:
                total_prom += int(match.group(1))
                successful_evals += 1
        avg_prom_score = (total_prom / successful_evals) if successful_evals > 0 else 0.0

    df = pd.DataFrame([{"ID": i, "Actual": "Fake" if t == 1 else "Real", "Predicted": "Fake" if p == 1 else "Real", "Model Output": pt} for i, (t, p, pt) in enumerate(zip(y_true, y_pred, [item['prediction']['text'] for item in results]))])
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    with open(OUTPUT_MD, "w", encoding='utf-8') as f:
        f.write(f"# Evaluation Report\nAccuracy: {acc*100:.2f}%\nPrecision: {prec*100:.2f}%\nRecall: {rec*100:.2f}%\nF1-Score: {f1*100:.2f}%\n\nBERTScore: {avg_bert_f1:.4f}\nPrometheus: {avg_prom_score:.2f} / 5.0")
    print(f"\n✅ Report saved to {OUTPUT_MD}")