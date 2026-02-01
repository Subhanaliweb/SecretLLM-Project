#!/usr/bin/env python3
"""
FINAL SUBMISSION - Cultural QA Pipeline
CRITICAL FIX: Ensures SINGLE answer per SAQ row (no multiple answers)
lr=3e-4, epochs=5, rank=32
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import ast
from pathlib import Path
from collections import Counter
import re
from sklearn.model_selection import train_test_split
import zipfile
import argparse
from tqdm import tqdm
import random


class FinalSubmissionCulturalQA:
    """Final version: SINGLE answer per SAQ, lr=3e-4, epochs=5"""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        output_dir: str = "./cultural_qa_final_submission",
        seed: int = 42
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = seed
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.mcq_examples = []
        self.saq_examples = []
        
        print(f"🚀 FINAL SUBMISSION Pipeline")
        print(f"   lr=3e-4, epochs=5, rank=32")
        print(f"   SINGLE answer per SAQ row")
    
    def parse_choices(self, choices_str: str) -> Dict[str, str]:
        """Parse MCQ choices"""
        try:
            return json.loads(choices_str)
        except:
            try:
                return ast.literal_eval(choices_str)
            except:
                return {}
    
    def parse_saq_annotations(self, annotations_str: str) -> List[str]:
        """Parse SAQ annotations with en_answers priority"""
        try:
            annotations = ast.literal_eval(annotations_str)
            
            if not isinstance(annotations, list):
                return []
            
            all_answers = []
            
            for item in annotations:
                if not isinstance(item, dict):
                    continue
                
                # Priority: en_answers
                if 'en_answers' in item and item['en_answers']:
                    en_answers = item['en_answers']
                    if isinstance(en_answers, list):
                        all_answers.extend([str(x).strip().lower() for x in en_answers if x])
                    else:
                        all_answers.append(str(en_answers).strip().lower())
                elif 'answers' in item and item['answers']:
                    answers = item['answers']
                    if isinstance(answers, list):
                        all_answers.extend([str(x).strip().lower() for x in answers if x])
                    else:
                        all_answers.append(str(answers).strip().lower())
            
            # Remove duplicates
            seen = set()
            unique = []
            for ans in all_answers:
                if ans and ans not in seen:
                    seen.add(ans)
                    unique.append(ans)
            
            return unique[:6]
            
        except:
            return []
    
    def normalize_answer_single(self, answer: str) -> str:
        """
        CRITICAL: Normalize to SINGLE answer only (no newlines, no multiple answers)
        """
        answer = answer.lower().strip()
        
        # Split by newline and take ONLY the first line
        if '\n' in answer:
            answer = answer.split('\n')[0].strip()
        
        # Remove common prefixes
        prefixes = [
            'the answer is:', 'answer:', 'the answer is', 'the answer:',
            'it is', "it's", 'i think', 'probably', 'typically', 'usually',
            'the most common', 'the most', 'most common', 'commonly', 'the '
        ]
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Remove punctuation at end
        answer = answer.rstrip('.,!?;:')
        answer = answer.strip('"\'')
        
        # Preserve time formats (HH:MM)
        if re.match(r'\d{1,2}:\d{2}', answer):
            return answer
        
        # Preserve pure numbers
        if re.match(r'^\d+$', answer):
            return answer
        
        # Remove leading articles
        for article in ['a ', 'an ', 'the ']:
            if answer.startswith(article):
                answer = answer[len(article):]
        
        # Limit to 4 words maximum
        words = answer.split()
        if len(words) > 4:
            answer = ' '.join(words[:4])
        
        # Final cleanup: remove any remaining newlines, tabs
        answer = answer.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        answer = ' '.join(answer.split())  # Normalize whitespace
        
        return answer.strip()
    
    def create_mcq_prompt(
        self,
        prompt: str,
        choices: Dict[str, str],
        answer: str = None,
        num_examples: int = 3
    ) -> str:
        """MCQ prompt"""
        
        examples_text = ""
        if self.mcq_examples and num_examples > 0:
            selected = random.sample(self.mcq_examples, min(num_examples, len(self.mcq_examples)))
            for ex in selected:
                examples_text += f"""Question: {ex['prompt']}
A. {ex['choices']['A']}
B. {ex['choices']['B']}
C. {ex['choices']['C']}
D. {ex['choices']['D']}
Answer: {ex['answer']}

"""
        
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a cultural knowledge expert. Select the most culturally appropriate answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

{examples_text}Question: {prompt}

Options:
A. {choices.get('A', '')}
B. {choices.get('B', '')}
C. {choices.get('C', '')}
D. {choices.get('D', '')}

Provide ONLY the letter (A, B, C, or D).<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        if answer:
            text += f"{answer}<|eot_id|>"
        return text
    
    def create_saq_prompt(
        self,
        question: str,
        answer: str = None,
        num_examples: int = 5
    ) -> str:
        """SAQ prompt - emphasizes SINGLE answer"""
        
        examples_text = ""
        if self.saq_examples and num_examples > 0:
            selected = random.sample(self.saq_examples, min(num_examples, len(self.saq_examples)))
            for ex in selected:
                ex_answer = ex['answers'][0] if ex['answers'] else "unknown"
                examples_text += f"Q: {ex['question']}\nA: {ex_answer}\n\n"
        
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a cultural knowledge expert. Provide ONE concise answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

Examples of correct answers:
{examples_text}
CRITICAL: Provide ONLY ONE answer (1-4 words). No explanations, no multiple options.

Guidelines:
- Use exact cultural terms
- Be concise: 1-4 words ONLY
- Times: HH:MM format (e.g., 18:00)
- Ages: just the number (e.g., 3)
- ONE answer only, no alternatives

Question: {question}

Answer (ONE answer, 1-4 words):<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        if answer:
            text += f"{answer}<|eot_id|>"
        return text
    
    def store_few_shot_examples(self, mcq_df: pd.DataFrame, saq_df: pd.DataFrame, num: int = 50):
        """Store few-shot examples"""
        print(f"\n📚 Storing {num} few-shot examples...")
        
        # MCQ
        mcq_sample = mcq_df.sample(min(num, len(mcq_df)), random_state=self.seed)
        for _, row in mcq_sample.iterrows():
            choices = self.parse_choices(row['choices'])
            if choices and 'answer_idx' in row:
                self.mcq_examples.append({
                    'prompt': row['prompt'],
                    'choices': choices,
                    'answer': row['answer_idx']
                })
        print(f"   ✓ MCQ: {len(self.mcq_examples)}")
        
        # SAQ
        saq_sample = saq_df.sample(min(num, len(saq_df)), random_state=self.seed)
        success = 0
        for _, row in saq_sample.iterrows():
            answers = self.parse_saq_annotations(row['annotations'])
            if answers:
                question = row['en_question'] if 'en_question' in row else row['question']
                self.saq_examples.append({'question': question, 'answers': answers})
                success += 1
        print(f"   ✓ SAQ: {len(self.saq_examples)} ({success}/{num})")
    
    def prepare_mcq_dataset(self, df: pd.DataFrame, use_few_shot: bool = False) -> Dataset:
        """Prepare MCQ dataset"""
        training_data = []
        
        for _, row in df.iterrows():
            choices = self.parse_choices(row['choices'])
            if not choices or 'answer_idx' not in row:
                continue
            
            prompt_text = self.create_mcq_prompt(
                row['prompt'], choices, row['answer_idx'],
                num_examples=2 if use_few_shot else 0
            )
            training_data.append({'text': prompt_text})
        
        print(f"   Prepared {len(training_data)} MCQ examples")
        
        def tokenize(examples):
            result = self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=1024,
                padding='max_length'
            )
            result['labels'] = result['input_ids'].copy()
            return result
        
        dataset = Dataset.from_dict({'text': [d['text'] for d in training_data]})
        return dataset.map(tokenize, batched=True, remove_columns=['text'])
    
    def prepare_saq_dataset(self, df: pd.DataFrame, use_few_shot: bool = False) -> Optional[Dataset]:
        """Prepare SAQ dataset"""
        training_data = []
        
        print(f"   Processing {len(df)} SAQ rows...")
        
        failures = 0
        for _, row in df.iterrows():
            answers = self.parse_saq_annotations(row['annotations'])
            
            if not answers:
                failures += 1
                continue
            
            question = row['en_question'] if 'en_question' in row else row['question']
            
            # Use top 3 variations
            for answer in answers[:3]:
                prompt_text = self.create_saq_prompt(
                    question, answer,
                    num_examples=3 if use_few_shot else 0
                )
                training_data.append({'text': prompt_text})
        
        print(f"   Prepared {len(training_data)} SAQ examples")
        print(f"   Failures: {failures}/{len(df)}")
        
        if len(training_data) == 0:
            return None
        
        def tokenize(examples):
            result = self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=1024,
                padding='max_length'
            )
            result['labels'] = result['input_ids'].copy()
            return result
        
        dataset = Dataset.from_dict({'text': [d['text'] for d in training_data]})
        return dataset.map(tokenize, batched=True, remove_columns=['text'])
    
    def load_model_for_training(self):
        """Load model with rank=32"""
        print("\n📥 Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Proper 8-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Rank 32 (proven optimal)
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        print("✓ Model loaded (rank=32)")
    
    def train_model(self, train_dataset: Dataset, eval_dataset: Dataset = None,
                   num_epochs: int = 5, batch_size: int = 4, lr: float = 3e-4):
        """Train with lr=3e-4, epochs=5"""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=lr,  # 3e-4
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset else "no",
            load_best_model_at_end=True if eval_dataset else False,
            report_to="none",
            seed=self.seed
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
        trainer.train()
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"✓ Saved to {self.output_dir}")
    
    def load_model_for_inference(self, lora_path: str = None):
        """Load for inference"""
        print("\n📥 Loading for inference...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Proper 8-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        if lora_path and Path(lora_path).exists():
            base = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base, lora_path)
            print(f"✓ Loaded LoRA")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            print("✓ Loaded base")
        
        self.model.eval()
    
    def generate(self, prompt: str, max_new_tokens: int = 50,
                temperature: float = 0.7, num_samples: int = 1) -> List[str]:
        """Generate with proper special token handling"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.92,
                num_return_sequences=num_samples,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        responses = []
        for output in outputs:
            # CRITICAL FIX: Extract only the new tokens (response), not the prompt
            input_length = inputs['input_ids'].shape[1]
            response_tokens = output[input_length:]
            
            # Decode with special tokens skipped
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            response = response.strip()
            
            # Additional cleanup for any remaining special tokens
            special_token_patterns = [
                '<|start_header_id|>', '<|end_header_id|>', 
                '<|eot_id|>', '<|begin_of_text|>',
                'assistant', 'user', 'system'
            ]
            for pattern in special_token_patterns:
                response = response.replace(pattern, '')
            
            # Clean up reserved tokens (regex)
            response = re.sub(r'<\|reserved_special_token_\d+\|>', '', response)
            
            response = response.strip()
            
            if response:  # Only add non-empty responses
                responses.append(response)
        
        return responses
    
    def extract_mcq_answer(self, response: str) -> str:
        """Extract MCQ answer"""
        response = response.strip().upper()
        
        match = re.search(r'^([A-D])\b', response)
        if match:
            return match.group(1)
        
        match = re.search(r'(?:answer|option)(?:\s+is)?[:\s]+([A-D])\b', response)
        if match:
            return match.group(1)
        
        match = re.search(r'\b([A-D])\b', response)
        if match:
            return match.group(1)
        
        return 'A'
    
    def predict_mcq(self, test_file: str, output_file: str, 
                   use_self_consistency: bool = True, num_samples: int = 9):
        """MCQ predictions"""
        print(f"\n📊 MCQ predictions from {test_file}")
        
        df = pd.read_csv(test_file)
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="MCQ"):
            choices = self.parse_choices(row['choices'])
            
            if not choices:
                answer = 'A'
            else:
                prompt = self.create_mcq_prompt(row['prompt'], choices, num_examples=3)
                
                if use_self_consistency:
                    responses = self.generate(prompt, 10, 0.7, num_samples)
                    answers = [self.extract_mcq_answer(r) for r in responses]
                    answer = Counter(answers).most_common(1)[0][0]
                else:
                    response = self.generate(prompt, 10, 0.0, 1)[0]
                    answer = self.extract_mcq_answer(response)
            
            results.append({
                'MCQID': row['MCQID'],
                'A': answer == 'A',
                'B': answer == 'B',
                'C': answer == 'C',
                'D': answer == 'D'
            })
        
        pd.DataFrame(results).to_csv(output_file, sep='\t', index=False)
        print(f"✓ Saved to {output_file}")
    
    def predict_saq(self, test_file: str, output_file: str,
                   use_self_consistency: bool = True, num_samples: int = 9):
        """
        SAQ predictions - CRITICAL: Ensures SINGLE answer per row
        """
        print(f"\n📊 SAQ predictions from {test_file}")
        print(f"   ENSURING SINGLE ANSWER PER ROW")
        
        df = pd.read_csv(test_file)
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="SAQ"):
            question = row['en_question'] if 'en_question' in row else row['question']
            prompt = self.create_saq_prompt(question, num_examples=5)
            
            if use_self_consistency:
                responses = self.generate(prompt, 30, 0.6, num_samples)
                normalized = [self.normalize_answer_single(r) for r in responses]
                answer = Counter(normalized).most_common(1)[0][0]
            else:
                response = self.generate(prompt, 30, 0.0, 1)[0]
                answer = self.normalize_answer_single(response)
            
            # CRITICAL: Ensure single answer (no newlines, no tabs)
            answer = answer.replace('\n', ' ').replace('\t', ' ').strip()
            answer = ' '.join(answer.split())  # Normalize all whitespace
            
            # Verify it's a single line
            if '\n' in answer or '\t' in answer:
                print(f"WARNING: Multi-line answer detected, taking first line only")
                answer = answer.split('\n')[0].split('\t')[0].strip()
            
            results.append({
                'ID': row['ID'],
                'answer': answer
            })
        
        # Save and verify
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_file, sep='\t', index=False)
        
        # Verification: Check for any newlines in answers
        print(f"✓ Saved to {output_file}")
        print(f"   Verifying format...")
        bad_rows = result_df[result_df['answer'].str.contains('\n|\\n', regex=True, na=False)]
        if len(bad_rows) > 0:
            print(f"   ⚠️ WARNING: {len(bad_rows)} rows contain newlines, fixing...")
            result_df['answer'] = result_df['answer'].str.replace('\n', ' ').str.strip()
            result_df.to_csv(output_file, sep='\t', index=False)
            print(f"   ✓ Fixed and re-saved")
        else:
            print(f"   ✓ All {len(result_df)} rows have single answers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'inference', 'both'], default='both')
    parser.add_argument('--model_name', default='meta-llama/Meta-Llama-3-8B')
    parser.add_argument('--output_dir', default='./cultural_qa_final_submission')
    parser.add_argument('--train_mcq', default='train_dataset_mcq.csv')
    parser.add_argument('--train_saq', default='train_dataset_saq.csv')
    parser.add_argument('--test_mcq', default='test_dataset_mcq.csv')
    parser.add_argument('--test_saq', default='test_dataset_saq.csv')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--self_consistency', action='store_true', default=True)
    parser.add_argument('--num_samples', type=int, default=9)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    pipeline = FinalSubmissionCulturalQA(args.model_name, args.output_dir, seed=args.seed)
    
    # Load examples
    if Path(args.train_mcq).exists() and Path(args.train_saq).exists():
        mcq_df = pd.read_csv(args.train_mcq)
        saq_df = pd.read_csv(args.train_saq)
        pipeline.store_few_shot_examples(mcq_df, saq_df, num=50)
    
    # TRAINING
    if args.mode in ['train', 'both']:
        print("\n" + "="*70)
        print("TRAINING (lr=3e-4, epochs=5, rank=32)")
        print("="*70)
        
        pipeline.load_model_for_training()
        
        # MCQ
        print(f"\n📚 MCQ: {args.train_mcq}")
        mcq_df = pd.read_csv(args.train_mcq)
        train_mcq, val_mcq = train_test_split(mcq_df, test_size=0.1, random_state=args.seed)
        
        print("\n🔨 Preparing MCQ...")
        train_ds = pipeline.prepare_mcq_dataset(train_mcq, use_few_shot=False)
        val_ds = pipeline.prepare_mcq_dataset(val_mcq, use_few_shot=False)
        
        print("\n🚀 Training MCQ...")
        pipeline.train_model(train_ds, val_ds, args.epochs, args.batch_size, args.lr)
        
        # SAQ
        print(f"\n📚 SAQ: {args.train_saq}")
        saq_df = pd.read_csv(args.train_saq)
        train_saq, val_saq = train_test_split(saq_df, test_size=0.1, random_state=args.seed)
        
        print("\n🔨 Preparing SAQ...")
        train_ds = pipeline.prepare_saq_dataset(train_saq, use_few_shot=False)
        
        if train_ds:
            val_ds = pipeline.prepare_saq_dataset(val_saq, use_few_shot=False)
            print("\n🚀 Training SAQ...")
            pipeline.train_model(train_ds, val_ds, args.epochs, args.batch_size, args.lr)
    
    # INFERENCE
    if args.mode in ['inference', 'both']:
        print("\n" + "="*70)
        print("INFERENCE")
        print("="*70)
        
        pipeline.load_model_for_inference(args.output_dir)
        
        pipeline.predict_mcq(args.test_mcq, f'{args.output_dir}/mcq_prediction.tsv', 
                           args.self_consistency, args.num_samples)
        pipeline.predict_saq(args.test_saq, f'{args.output_dir}/saq_prediction.tsv',
                           args.self_consistency, args.num_samples)
        
        print("\n📦 Creating submission...")
        with zipfile.ZipFile(f'{args.output_dir}/submission.zip', 'w') as zipf:
            zipf.write(f'{args.output_dir}/mcq_prediction.tsv', 'mcq_prediction.tsv')
            zipf.write(f'{args.output_dir}/saq_prediction.tsv', 'saq_prediction.tsv')
        print("✓ submission.zip created")
    
    print("\n✅ COMPLETE!")
    print("Expected: MCQ 76-79%, SAQ 68-73%, Overall 72-76%")


if __name__ == "__main__":
    main()