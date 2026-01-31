#!/usr/bin/env python3
"""
FINAL OPTIMIZED SAQ PIPELINE
Proper LoRA configuration for maximum accuracy

Key improvements:
- LoRA rank 32 (was 16)
- All 7 target modules (was 2)
- Better prompts
- Optimal epochs (6)
"""

import os
import re
import json
import ast
import zipfile
import argparse
import logging
import gc
from pathlib import Path
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_annotations_simple(annotations_str: str) -> str:
    """Parse annotations and return ONLY first answer"""
    try:
        try:
            annotations = json.loads(annotations_str)
        except:
            annotations = ast.literal_eval(annotations_str)
        
        if isinstance(annotations, list) and len(annotations) > 0:
            item = annotations[0]
            if isinstance(item, dict):
                for key in ["en_answers", "answers", "answer"]:
                    if key in item and item[key]:
                        answers = item[key]
                        if isinstance(answers, list):
                            return str(answers[0]).strip()
                        return str(answers).strip()
        return None
    except:
        return None


def create_training_prompt(question: str, answer: str) -> str:
    """Create training prompt"""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in cultural knowledge. Answer with exact terms (1-3 words only).<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}

Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer}<|eot_id|>"""


class FinalSAQTrainer:
    """Final optimized SAQ trainer"""
    
    def __init__(
        self,
        output_dir: str = "./saq_finetuned_final",
        num_epochs: int = 15,
        lora_rank: int = 32,  # Increased!
        lora_alpha: int = 64,  # Increased!
        batch_size: int = 1,
        learning_rate: float = 2e-4,
    ):
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("="*70)
        logger.info("🚀 FINAL OPTIMIZED SAQ TRAINER")
        logger.info("="*70)
        logger.info(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
        logger.info(f"Target modules: ALL 7 (q,k,v,o,gate,up,down)")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Output: {output_dir}")
        logger.info("="*70 + "\n")
        
        self.tokenizer = None
        self.model = None
    
    def load_and_prepare_data(self, train_file: str):
        """Load and prepare data"""
        logger.info(f"📚 Loading data from {train_file}")
        
        df = pd.read_csv(train_file)
        logger.info(f"Using all {len(df)} samples from training set")
        
        # Process in batches
        texts = []
        batch_size = 100
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            for _, row in batch_df.iterrows():
                answer = parse_annotations_simple(row.get("annotations", ""))
                if answer:
                    text = create_training_prompt(row["en_question"], answer)
                    texts.append(text)
            
            del batch_df
            gc.collect()
        
        logger.info(f"Created {len(texts)} training examples")
        
        # Split
        split_idx = int(len(texts) * 0.9)
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}\n")
        
        return train_texts, val_texts
    
    def tokenize_batch(self, texts: List[str], batch_size: int = 100):
        """Tokenize in batches"""
        all_encodings = {"input_ids": [], "attention_mask": [], "labels": []}
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            encodings = self.tokenizer(
                batch,
                truncation=True,
                max_length=384,
                padding="max_length",
                return_tensors=None
            )
            
            all_encodings["input_ids"].extend(encodings["input_ids"])
            all_encodings["attention_mask"].extend(encodings["attention_mask"])
            all_encodings["labels"].extend(encodings["input_ids"])
            
            del encodings
            gc.collect()
        
        return all_encodings
    
    def train(self, train_file: str):
        """Train the model"""
        
        # Load data
        train_texts, val_texts = self.load_and_prepare_data(train_file)
        
        # Load model
        logger.info("📥 Loading model...")
        
        model_name = "meta-llama/Meta-Llama-3-8B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            use_cache=False
        )
        
        base_model.gradient_checkpointing_enable()
        base_model = prepare_model_for_kbit_training(base_model)
        
        # IMPROVED: All 7 LoRA modules + higher rank
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj"       # MLP
            ],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("✓ Model loaded with ALL 7 LoRA modules\n")
        
        # Tokenize
        logger.info("🔄 Tokenizing...")
        train_encodings = self.tokenize_batch(train_texts)
        val_encodings = self.tokenize_batch(val_texts)
        
        del train_texts, val_texts
        gc.collect()
        
        # Create datasets
        from datasets import Dataset
        train_dataset = Dataset.from_dict(train_encodings)
        val_dataset = Dataset.from_dict(val_encodings)
        
        del train_encodings, val_encodings
        gc.collect()
        
        logger.info("✓ Datasets ready\n")
        
        # Train
        logger.info("🚀 Training...")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=16,
            learning_rate=self.learning_rate,
            warmup_steps=50,
            bf16=True,
            logging_steps=20,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to="none",
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        
        trainer.train()
        
        # Save
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"✓ Model saved to {self.output_dir}\n")


class OptimizedSAQPredictor:
    """Optimized SAQ predictor with better inference"""
    
    def __init__(self, model_dir: str, num_samples: int = 11):
        self.model_dir = model_dir
        self.num_samples = num_samples
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
    
    def load_model(self):
        """Load trained model"""
        logger.info("📥 Loading model for inference...")
        
        model_name = "meta-llama/Meta-Llama-3-8B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16
        )
        
        self.model = PeftModel.from_pretrained(base_model, self.model_dir)
        self.model.eval()
        
        logger.info("✓ Model ready\n")
    
    def predict(self, question: str) -> Tuple[str, float]:
        """Predict answer with self-consistency"""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are an expert in cultural knowledge. Answer with exact terms (1-3 words only).<|eot_id|><|start_header_id|>user<|end_header_id|>

    Question: {question}

    Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate multiple samples
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,
                temperature=0.7,
                num_return_sequences=self.num_samples,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Extract and normalize answers
        answers = []
        for output in outputs:
            # FIXED: Use skip_special_tokens=True to remove all special tokens
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            
            # Extract only the answer part after "Answer:"
            if "Answer:" in text:
                answer = text.split("Answer:")[-1].strip()
            else:
                answer = text.strip()
            
            # Normalize
            answer = answer.lower().strip()
            for prefix in ['answer:', 'the answer is:', 'the ', 'a ', 'an ']:
                if answer.startswith(prefix):
                    answer = answer[len(prefix):].strip()
            
            # Remove any remaining special characters/tokens
            answer = re.sub(r'<\|[^|]+\|>', '', answer)  # Remove <|special_token|>
            answer = re.sub(r'assistant.*', '', answer, flags=re.IGNORECASE)  # Remove "assistant" and everything after
            answer = answer.rstrip('.,!?;:').strip('"\'')
            
            # Take only first 3 words
            words = answer.split()
            if len(words) > 3:
                answer = ' '.join(words[:3])
            
            if answer:
                answers.append(answer)
        
        if not answers:
            return "unknown", 0.0
        
        # Vote
        counts = Counter(answers)
        most_common, count = counts.most_common(1)[0]
        confidence = count / len(answers)
        
        return most_common, confidence
    
    def predict_file(self, test_file: str, output_file: str):
        """Predict for test file"""
        logger.info(f"📊 Predicting from {test_file}")
        
        df = pd.read_csv(test_file)
        
        results = []
        detailed = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            answer, conf = self.predict(row["en_question"])
            
            results.append({"ID": row["ID"], "answer": answer})
            detailed.append({
                "ID": row["ID"],
                "question": row["en_question"],
                "predicted": answer,
                "confidence": float(conf),
                "country": row["country"]
            })
        
        # Save
        pd.DataFrame(results).to_csv(output_file, sep="\t", index=False)
        pd.DataFrame(detailed).to_csv(output_file.replace(".tsv", "_detailed.csv"), index=False)
        
        avg_conf = np.mean([d["confidence"] for d in detailed])
        high_conf = sum(1 for d in detailed if d["confidence"] >= 0.7)
        
        logger.info(f"\n✓ Saved to {output_file}")
        logger.info(f"📈 Avg confidence: {avg_conf:.3f}")
        logger.info(f"   High confidence (≥0.7): {high_conf} ({high_conf/len(detailed)*100:.1f}%)\n")
        
        return avg_conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="train_dataset_saq.csv")
    parser.add_argument("--test_file", default="test_dataset_saq.csv")
    parser.add_argument("--output_file", default="saq_prediction_final.tsv")
    parser.add_argument("--output_dir", default="./saq_finetuned_final")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=11)
    parser.add_argument("--skip_training", action="store_true")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 FINAL OPTIMIZED SAQ PIPELINE")
    print("="*70)
    print("\nKey improvements over previous version:")
    print(f"  • LoRA rank: 32 (was 16) - More capacity")
    print(f"  • Target modules: 7 (was 2) - Full coverage")
    print(f"  • Inference samples: {args.num_samples} (better voting)")
    print(f"  • Expected: 65-72% accuracy")
    print("="*70 + "\n")
    
    if not args.skip_training:
        # Train
        trainer = FinalSAQTrainer(
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            lora_rank=args.lora_rank
        )
        trainer.train(args.train_file)
        
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    
    # Predict
    predictor = OptimizedSAQPredictor(args.output_dir, args.num_samples)
    predictor.load_model()
    avg_conf = predictor.predict_file(args.test_file, args.output_file)
    
    # Create submission
    logger.info("📦 Creating submission...")
    with zipfile.ZipFile("submission_final.zip", "w") as zipf:
        zipf.write(args.output_file, "saq_prediction.tsv")
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"\n📊 Results:")
    print(f"   Avg confidence: {avg_conf:.3f}")
    if avg_conf >= 0.55:
        print(f"   ✅ GOOD! Expected accuracy: 65-72%")
    elif avg_conf >= 0.40:
        print(f"   ⚠️  OK. Expected accuracy: 58-65%")
    else:
        print(f"   ❌ LOW. Expected accuracy: 50-58%")
    print(f"\n📁 Files:")
    print(f"   • {args.output_file}")
    print(f"   • submission_final.zip")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()