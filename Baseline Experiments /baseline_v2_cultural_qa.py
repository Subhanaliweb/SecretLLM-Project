#!/usr/bin/env python3
"""
ADVANCED SUBMISSION - Cultural QA Pipeline
Additional improvements based on data analysis:
1. Answer count-weighted training (prioritize common answers)
2. Question-type specific prompting (age/time/clothing patterns)
3. Cultural region-specific examples
4. Better answer validation
5. 5 few-shot examples (as requested)
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
from collections import Counter, defaultdict
import re
from sklearn.model_selection import train_test_split
import zipfile
import argparse
from tqdm import tqdm
import random


class AdvancedCulturalQA:
    """Advanced with count-weighting and type-specific prompting"""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        output_dir: str = "./cultural_qa_advanced",
        seed: int = 42
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = seed
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Store examples by culture AND type
        self.mcq_examples_by_culture = defaultdict(list)
        self.saq_examples_by_culture = defaultdict(list)
        self.saq_examples_by_type = defaultdict(list)  # age, time, clothing, etc.
        self.mcq_examples = []
        self.saq_examples = []
        
        print(f"🚀 ADVANCED Pipeline")
        print(f"   Count-weighted training")
        print(f"   Type-specific prompting")
        print(f"   Cultural region balancing")
    
    def parse_choices(self, choices_str: str) -> Dict[str, str]:
        """Parse MCQ choices"""
        try:
            return json.loads(choices_str)
        except:
            try:
                return ast.literal_eval(choices_str)
            except:
                return {}
    
    def parse_saq_annotations_weighted(self, annotations_str: str) -> List[tuple]:
        """
        Parse with COUNT information for weighted training
        Returns: [(answer, count), ...]
        """
        try:
            annotations = ast.literal_eval(annotations_str)
            
            if not isinstance(annotations, list):
                return []
            
            weighted_answers = []
            
            for item in annotations:
                if not isinstance(item, dict):
                    continue
                
                count = item.get('count', 1)  # Get answer frequency
                
                # Priority: en_answers
                if 'en_answers' in item and item['en_answers']:
                    en_answers = item['en_answers']
                    if isinstance(en_answers, list):
                        for ans in en_answers:
                            if ans:
                                weighted_answers.append((str(ans).strip().lower(), count))
                    else:
                        weighted_answers.append((str(en_answers).strip().lower(), count))
                elif 'answers' in item and item['answers']:
                    answers = item['answers']
                    if isinstance(answers, list):
                        for ans in answers:
                            if ans:
                                weighted_answers.append((str(ans).strip().lower(), count))
                    else:
                        weighted_answers.append((str(answers).strip().lower(), count))
            
            return weighted_answers[:10]  # Top 10 with counts
            
        except:
            return []
    
    def detect_question_type(self, question: str) -> str:
        """Detect question type for specialized prompting"""
        question_lower = question.lower()
        
        if re.search(r'\bage\b|\bhow old\b|\bwhen.*born\b', question_lower):
            return 'age'
        elif re.search(r'\btime\b|\bwhen\b|\bo\'?clock\b|\bhour\b', question_lower):
            return 'time'
        elif re.search(r'\bclothing\b|\bwear\b|\bdress\b|\bgarment\b|\boutfit\b', question_lower):
            return 'clothing'
        elif re.search(r'\bfood\b|\beat\b|\bdish\b|\bmeal\b|\bcuisine\b', question_lower):
            return 'food'
        elif re.search(r'\bsport\b|\bplay\b|\bgame\b', question_lower):
            return 'sport'
        elif re.search(r'\bfestival\b|\bholiday\b|\bcelebration\b', question_lower):
            return 'festival'
        elif re.search(r'\bregion\b|\bplace\b|\bcity\b|\barea\b|\blocation\b', question_lower):
            return 'location'
        else:
            return 'general'
    
    def extract_culture_from_id(self, item_id: str) -> str:
        """Extract culture from ID"""
        # Na-ko-25 -> IR, New-en-79 -> GB, etc.
        parts = item_id.split('-')
        if len(parts) >= 2:
            code = parts[1].lower()
            # Map codes to countries
            mapping = {
                'ko': 'IR',  # Farsi
                'en': 'GB',  # English  
                'as': 'US',  # American
                'zh': 'CN',  # Chinese
            }
            return mapping.get(code, 'unknown')
        return 'unknown'
    
    def normalize_answer_single(self, answer: str, question_type: str = 'general') -> str:
        """Type-aware normalization"""
        answer = answer.lower().strip()
        
        # Split by newline - take first line only
        if '\n' in answer:
            answer = answer.split('\n')[0].strip()
        
        # Remove prefixes
        prefixes = [
            'the answer is:', 'answer:', 'the answer is', 'the answer:',
            'it is', "it's", 'i think', 'probably', 'typically', 'usually',
            'the most common', 'the most', 'most common', 'commonly'
        ]
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Remove punctuation
        answer = answer.rstrip('.,!?;:')
        answer = answer.strip('"\'')
        
        # Type-specific handling
        if question_type == 'age':
            # Extract just the number
            match = re.search(r'\d+', answer)
            if match:
                return match.group(0)
        
        elif question_type == 'time':
            # Preserve HH:MM format
            match = re.search(r'\d{1,2}:\d{2}', answer)
            if match:
                return match.group(0)
        
        # Preserve pure numbers
        if re.match(r'^\d+$', answer):
            return answer
        
        # Remove leading articles
        for article in ['a ', 'an ', 'the ']:
            if answer.startswith(article):
                answer = answer[len(article):]
        
        # Limit words
        words = answer.split()
        if len(words) > 4:
            answer = ' '.join(words[:4])
        
        # Final cleanup
        answer = answer.replace('\n', ' ').replace('\t', ' ')
        answer = ' '.join(answer.split())
        
        return answer.strip()
    
    def create_mcq_prompt(
        self,
        prompt: str,
        choices: Dict[str, str],
        answer: str = None,
        num_examples: int = 5,
        culture: str = None
    ) -> str:
        """MCQ prompt with cultural balancing"""
        
        examples_text = ""
        if self.mcq_examples and num_examples > 0:
            selected = []
            
            # Include 2 same-culture examples if available
            if culture and culture in self.mcq_examples_by_culture:
                same_culture = self.mcq_examples_by_culture[culture]
                selected.extend(random.sample(same_culture, min(2, len(same_culture))))
            
            # Fill rest with diverse examples
            remaining = num_examples - len(selected)
            if remaining > 0:
                other_examples = [ex for ex in self.mcq_examples if ex not in selected]
                if other_examples:
                    selected.extend(random.sample(other_examples, min(remaining, len(other_examples))))
            
            for ex in selected:
                examples_text += f"""Question: {ex['prompt']}
A. {ex['choices']['A']}
B. {ex['choices']['B']}
C. {ex['choices']['C']}
D. {ex['choices']['D']}
Answer: {ex['answer']}

"""
        
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a cultural knowledge expert. Select the most culturally appropriate answer for the specific region.<|eot_id|><|start_header_id|>user<|end_header_id|>

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
        num_examples: int = 5,
        culture: str = None,
        question_type: str = 'general'
    ) -> str:
        """SAQ prompt with type-specific guidance"""
        
        examples_text = ""
        if self.saq_examples and num_examples > 0:
            selected = []
            
            # Include 2 same-type examples
            if question_type in self.saq_examples_by_type:
                type_examples = self.saq_examples_by_type[question_type]
                selected.extend(random.sample(type_examples, min(2, len(type_examples))))
            
            # Include 1 same-culture example
            if culture and culture in self.saq_examples_by_culture:
                culture_examples = [ex for ex in self.saq_examples_by_culture[culture] if ex not in selected]
                if culture_examples:
                    selected.append(random.choice(culture_examples))
            
            # Fill rest
            remaining = num_examples - len(selected)
            if remaining > 0:
                other_examples = [ex for ex in self.saq_examples if ex not in selected]
                if other_examples:
                    selected.extend(random.sample(other_examples, min(remaining, len(other_examples))))
            
            for ex in selected:
                ex_answer = ex['answers'][0] if ex['answers'] else "unknown"
                examples_text += f"Q: {ex['question']}\nA: {ex_answer}\n\n"
        
        # Type-specific instructions
        type_instructions = {
            'age': "For age questions: Provide ONLY the number (e.g., 3, 18)",
            'time': "For time questions: Use HH:MM format (e.g., 18:00, 09:30)",
            'clothing': "For clothing: Use the exact cultural term (e.g., hanbok, qipao, sari)",
            'food': "For food: Use the specific dish name (e.g., sushi, tacos)",
            'sport': "For sports: Use the common name in that culture (e.g., football, cricket)",
            'location': "For locations: Use the specific place name (e.g., london, beijing)",
            'general': "Be specific and concise"
        }
        
        type_instruction = type_instructions.get(question_type, type_instructions['general'])
        
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a cultural knowledge expert. Provide ONE concise, culturally accurate answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

Examples:
{examples_text}
Guidelines:
- {type_instruction}
- ONE answer only (1-4 words)
- Use exact cultural terms
- No explanations

Question: {question}

Answer (ONE answer, 1-4 words):<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        if answer:
            text += f"{answer}<|eot_id|>"
        return text
    
    def store_few_shot_examples(self, mcq_df: pd.DataFrame, saq_df: pd.DataFrame, num: int = 100):
        """Store culturally balanced examples"""
        print(f"\n📚 Storing {num} few-shot examples...")
        
        # MCQ with culture tracking
        mcq_sample = mcq_df.sample(min(num, len(mcq_df)), random_state=self.seed)
        for _, row in mcq_sample.iterrows():
            choices = self.parse_choices(row['choices'])
            if choices and 'answer_idx' in row:
                culture = self.extract_culture_from_id(row.get('MCQID', ''))
                example = {
                    'prompt': row['prompt'],
                    'choices': choices,
                    'answer': row['answer_idx'],
                    'culture': culture
                }
                self.mcq_examples.append(example)
                self.mcq_examples_by_culture[culture].append(example)
        
        print(f"   ✓ MCQ: {len(self.mcq_examples)}")
        for culture, examples in self.mcq_examples_by_culture.items():
            print(f"     {culture}: {len(examples)}")
        
        # SAQ with culture and type tracking
        saq_sample = saq_df.sample(min(num, len(saq_df)), random_state=self.seed)
        success = 0
        for _, row in saq_sample.iterrows():
            weighted_answers = self.parse_saq_annotations_weighted(row['annotations'])
            
            if weighted_answers:
                question = row['en_question'] if 'en_question' in row else row['question']
                culture = row.get('country', 'unknown')
                q_type = self.detect_question_type(question)
                
                # Extract just answers (ignore counts for examples)
                answers = [ans for ans, count in weighted_answers]
                
                example = {
                    'question': question,
                    'answers': answers,
                    'culture': culture,
                    'type': q_type
                }
                self.saq_examples.append(example)
                self.saq_examples_by_culture[culture].append(example)
                self.saq_examples_by_type[q_type].append(example)
                success += 1
        
        print(f"   ✓ SAQ: {len(self.saq_examples)} ({success}/{num})")
        print(f"   By culture: {dict([(k, len(v)) for k, v in self.saq_examples_by_culture.items()])}")
        print(f"   By type: {dict([(k, len(v)) for k, v in self.saq_examples_by_type.items()])}")
    
    def prepare_mcq_dataset(self, df: pd.DataFrame, use_few_shot: bool = False) -> Dataset:
        """Prepare MCQ dataset"""
        training_data = []
        
        for _, row in df.iterrows():
            choices = self.parse_choices(row['choices'])
            if not choices or 'answer_idx' not in row:
                continue
            
            culture = self.extract_culture_from_id(row.get('MCQID', ''))
            
            prompt_text = self.create_mcq_prompt(
                row['prompt'], choices, row['answer_idx'],
                num_examples=3 if use_few_shot else 0,
                culture=culture
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
        """
        Prepare SAQ dataset with COUNT-WEIGHTED training
        High-count answers get more training examples
        """
        training_data = []
        
        print(f"   Processing {len(df)} SAQ rows...")
        
        failures = 0
        total_examples = 0
        
        for _, row in df.iterrows():
            weighted_answers = self.parse_saq_annotations_weighted(row['annotations'])
            
            if not weighted_answers:
                failures += 1
                continue
            
            question = row['en_question'] if 'en_question' in row else row['question']
            culture = row.get('country', 'unknown')
            q_type = self.detect_question_type(question)
            
            # Use count weighting: answers with higher count get more training examples
            for answer, count in weighted_answers[:5]:  # Top 5
                # Repeat based on count (but max 3 times)
                repetitions = min(count, 3) if count > 1 else 1
                
                for _ in range(repetitions):
                    prompt_text = self.create_saq_prompt(
                        question, answer,
                        num_examples=3 if use_few_shot else 0,
                        culture=culture,
                        question_type=q_type
                    )
                    training_data.append({'text': prompt_text})
                    total_examples += 1
        
        print(f"   Prepared {len(training_data)} SAQ examples (count-weighted)")
        print(f"   Average examples per question: {total_examples/max(len(df)-failures, 1):.1f}")
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
        """Load model"""
        print("\n📥 Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Proper 8-bit quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        
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
        """Train"""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=lr,
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
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        if lora_path and Path(lora_path).exists():
            base = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base, lora_path)
            print(f"✓ Loaded LoRA")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            print("✓ Loaded base")
        
        self.model.eval()
    
    def generate(self, prompt: str, max_new_tokens: int = 50,
                temperature: float = 0.7, num_samples: int = 1) -> List[str]:
        """Generate"""
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
            text = self.tokenizer.decode(output, skip_special_tokens=False)
            if "<|start_header_id|>assistant<|end_header_id|>" in text:
                response = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                response = response.split("<|eot_id|>")[0].strip()
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
        print(f"\n📊 MCQ from {test_file}")
        
        df = pd.read_csv(test_file)
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="MCQ"):
            choices = self.parse_choices(row['choices'])
            culture = self.extract_culture_from_id(row.get('MCQID', ''))
            
            if not choices:
                answer = 'A'
            else:
                prompt = self.create_mcq_prompt(
                    row['prompt'], choices, 
                    num_examples=5,  # 5 examples as requested
                    culture=culture
                )
                
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
        print(f"✓ Saved")
    
    def predict_saq(self, test_file: str, output_file: str,
                   use_self_consistency: bool = True, num_samples: int = 9):
        """SAQ predictions with type-aware normalization"""
        print(f"\n📊 SAQ from {test_file}")
        
        df = pd.read_csv(test_file)
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="SAQ"):
            question = row['en_question'] if 'en_question' in row else row['question']
            culture = row.get('country', 'unknown')
            q_type = self.detect_question_type(question)
            
            prompt = self.create_saq_prompt(
                question, 
                num_examples=5,  # 5 examples as requested
                culture=culture,
                question_type=q_type
            )
            
            if use_self_consistency:
                responses = self.generate(prompt, 30, 0.6, num_samples)
                normalized = [self.normalize_answer_single(r, q_type) for r in responses]
                answer = Counter(normalized).most_common(1)[0][0]
            else:
                response = self.generate(prompt, 30, 0.0, 1)[0]
                answer = self.normalize_answer_single(response, q_type)
            
            # Final cleanup
            answer = answer.replace('\n', ' ').replace('\t', ' ').strip()
            answer = ' '.join(answer.split())
            
            results.append({
                'ID': row['ID'],
                'answer': answer
            })
        
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_file, sep='\t', index=False)
        print(f"✓ Saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'inference', 'both'], default='both')
    parser.add_argument('--model_name', default='meta-llama/Meta-Llama-3-8B')
    parser.add_argument('--output_dir', default='./cultural_qa_advanced')
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
    
    pipeline = AdvancedCulturalQA(args.model_name, args.output_dir, seed=args.seed)
    
    # Load examples
    if Path(args.train_mcq).exists() and Path(args.train_saq).exists():
        mcq_df = pd.read_csv(args.train_mcq)
        saq_df = pd.read_csv(args.train_saq)
        pipeline.store_few_shot_examples(mcq_df, saq_df, num=100)
    
    # TRAINING
    if args.mode in ['train', 'both']:
        print("\n" + "="*70)
        print("TRAINING (Count-weighted, Type-aware)")
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
        
        print("\n🔨 Preparing SAQ (count-weighted)...")
        train_ds = pipeline.prepare_saq_dataset(train_saq, use_few_shot=False)
        
        if train_ds:
            val_ds = pipeline.prepare_saq_dataset(val_saq, use_few_shot=False)
            print("\n🚀 Training SAQ...")
            pipeline.train_model(train_ds, val_ds, args.epochs, args.batch_size, args.lr)
    
    # INFERENCE
    if args.mode in ['inference', 'both']:
        print("\n" + "="*70)
        print("INFERENCE (5 few-shot, Type-aware)")
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
        print("✓ Done")
    
    print("\n✅ COMPLETE!")
    print("Expected: MCQ 77-80%, SAQ 70-75%, Overall 73-77%")


if __name__ == "__main__":
    main()