"""
All-in-One: Fine-tune + Predict MCQ
Just run: python mcq_finetune_predict.py
"""

import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import re
from tqdm import tqdm

# ==================== CONFIGURATION ====================
HF_TOKEN = "your_huggingface_token_here"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

print("="*70)
print("MCQ Fine-tuning + Prediction Pipeline")
print("="*70)

# ==================== LOAD DATA ====================
print("\n📁 Loading datasets...")
train_df = pd.read_csv('train_dataset_mcq.csv')
test_df = pd.read_csv('test_dataset_mcq.csv')
print(f"   Train: {len(train_df)} samples")
print(f"   Test: {len(test_df)} samples")

# ==================== INITIALIZE MODEL ====================
print("\n📦 Loading model with 4-bit quantization...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    token=HF_TOKEN
)

model = prepare_model_for_kbit_training(model)

# ==================== CONFIGURE LORA ====================
print("📦 Configuring LoRA...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==================== PREPARE TRAINING DATA ====================
def create_prompt(question, options, answer=None, country=None):
    country_hint = f" (Country: {country})" if country and country != "unknown" else ""
    
    prompt = f"""Answer the following multiple choice question about cultural knowledge{country_hint}.

Question: {question}

Options:
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

Answer: {answer if answer else ''}"""
    
    return prompt.strip()

print("\n📊 Preparing training data...")
train_texts = []
for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing"):
    prompt_text = row['prompt']
    lines = prompt_text.split('\n')
    question = lines[0] if lines else ""
    
    try:
        choices = json.loads(row['choices'])
        options = {'A': choices.get('A', ''), 'B': choices.get('B', ''), 'C': choices.get('C', ''), 'D': choices.get('D', '')}
    except:
        continue
    
    answer = row['answer_idx']
    country = row.get('country', None)
    
    full_text = create_prompt(question, options, answer, country)
    train_texts.append(full_text)

print(f"✓ Created {len(train_texts)} training examples")

# ==================== TOKENIZE ====================
print("🔧 Tokenizing...")
train_dataset = Dataset.from_dict({"text": train_texts})

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing"
)

# Split for validation
split = train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split['train']
val_dataset = split['test']

# ==================== TRAINING ====================
print("\n🚀 Starting fine-tuning...")

training_args = TrainingArguments(
    output_dir="./temp_model",
    num_train_epochs=20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=10,
    eval_steps=50,
    save_steps=1000,
    eval_strategy="steps",
    save_strategy="no",
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

print("✓ Fine-tuning complete!")

# ==================== INFERENCE ====================
print("\n" + "="*70)
print("Running Inference on Test Set")
print("="*70)

model.eval()
tokenizer.padding_side = "left"

def extract_answer(generated_text, prompt):
    response = generated_text[len(prompt):].strip()[:10]
    
    patterns = [
        r'^([A-D])\b',
        r'^([A-D])\)',
        r'^([A-D])\.',
        r'^\(([A-D])\)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    match = re.search(r'[A-D]', response.upper())
    if match:
        return match.group(0)
    
    return "A"

# Parse test data
print("\n📊 Processing test data...")
test_questions = []
test_options = []
test_ids = []
test_countries = []

for idx, row in test_df.iterrows():
    prompt_text = row['prompt']
    lines = prompt_text.split('\n')
    question = lines[0] if lines else ""
    
    try:
        choices = json.loads(row['choices'])
        options = {'A': choices.get('A', ''), 'B': choices.get('B', ''), 'C': choices.get('C', ''), 'D': choices.get('D', '')}
    except:
        options = {'A': '', 'B': '', 'C': '', 'D': ''}
    
    test_questions.append(question)
    test_options.append(options)
    test_ids.append(row['MCQID'])
    test_countries.append(row.get('country', None))

# Predict
print("🔮 Generating predictions...")
predictions = []
batch_size = 4

for i in tqdm(range(0, len(test_questions), batch_size), desc="Predicting"):
    batch_q = test_questions[i:i+batch_size]
    batch_opts = test_options[i:i+batch_size]
    batch_countries = test_countries[i:i+batch_size]
    
    prompts = [
        create_prompt(q, opts, answer=None, country=country)
        for q, opts, country in zip(batch_q, batch_opts, batch_countries)
    ]
    
    # Add "Answer:" at the end for generation
    prompts = [p + "\nAnswer:" for p in prompts]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    for prompt, output in zip(prompts, outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        answer = extract_answer(generated_text, prompt)
        predictions.append(answer)

# ==================== VALIDATE ====================
print("\n🧪 Validating on training sample...")
val_sample = train_df.sample(n=100, random_state=42)
val_predictions = []
val_questions = []
val_options = []

for idx, row in val_sample.iterrows():
    prompt_text = row['prompt']
    lines = prompt_text.split('\n')
    question = lines[0] if lines else ""
    
    try:
        choices = json.loads(row['choices'])
        options = {'A': choices.get('A', ''), 'B': choices.get('B', ''), 'C': choices.get('C', ''), 'D': choices.get('D', '')}
    except:
        continue
    
    country = row.get('country', None)
    prompt = create_prompt(question, options, answer=None, country=country) + "\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=3, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = extract_answer(generated_text, prompt)
    val_predictions.append(answer)

correct = sum(1 for i, (_, row) in enumerate(val_sample.iterrows()) if val_predictions[i] == row['answer_idx'])
val_accuracy = correct / len(val_predictions) * 100
print(f"Validation Accuracy: {correct}/{len(val_predictions)} = {val_accuracy:.2f}%")

# ==================== SAVE RESULTS ====================
print("\n💾 Saving predictions...")
submission_data = []
for mcq_id, pred in zip(test_ids, predictions):
    submission_data.append({
        'MCQID': mcq_id,
        'A': 'True' if pred == 'A' else 'False',
        'B': 'True' if pred == 'B' else 'False',
        'C': 'True' if pred == 'C' else 'False',
        'D': 'True' if pred == 'D' else 'False'
    })

submission_df = pd.DataFrame(submission_data)
submission_df.to_csv('mcq_prediction_finetuned.tsv', sep='\t', index=False)

print("\n" + "="*70)
print("✅ COMPLETE!")
print("="*70)
print(f"Validation Accuracy: {val_accuracy:.2f}%")
print(f"Test Predictions: {len(predictions)}")
print(f"Output: mcq_prediction_finetuned.tsv")
print("\nPrediction Distribution:")
print(pd.Series(predictions).value_counts())
print("\n" + "="*70)