#!/usr/bin/env python3
"""
CULTURAL QA RESEARCH PIPELINE (FIXED)
- Uses PEFT LoRA/QLoRA so Transformers Trainer can train on quantized models.
- Keeps your training/eval/inference pipeline structure.
"""

import os
import re
import json
import math
import time
import zipfile
import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)

# PEFT (IMPORTANT FIX)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

# Optional W&B integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  W&B not available. Install with: pip install wandb")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for ablation experiments"""

    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None

    # Training parameters
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100

    # Model parameters
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    max_length: int = 512

    # Experiment metadata
    experiment_name: str = "baseline"
    output_dir: str = "./experiments"
    seed: int = 42

    # Inference parameters
    use_self_consistency: bool = True
    num_samples: int = 5
    temperature: float = 0.7

    # Quantization (QLoRA)
    use_4bit: bool = True  # QLoRA
    bnb_4bit_quant_type: str = "nf4"
    bnb_double_quant: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # Always write into a per-experiment subfolder
        self.output_dir = f"{self.output_dir}/{self.experiment_name}"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ============================================================================
# TRAINING DYNAMICS TRACKER
# ============================================================================

class TrainingDynamicsTracker:
    """Track training dynamics for analysis"""

    def __init__(self):
        self.history = []
        self.start_time = time.time()

    def log_step(self, epoch: float, step: int, loss: float,
                 learning_rate: float, grad_norm: Optional[float] = None):
        self.history.append({
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
            "timestamp": time.time() - self.start_time
        })

    def get_summary(self) -> Dict:
        if not self.history:
            return {}
        df = pd.DataFrame(self.history)
        return {
            "total_steps": len(self.history),
            "avg_loss": float(df["loss"].mean()),
            "final_loss": float(df["loss"].iloc[-1]),
            "loss_std": float(df["loss"].std()),
            "training_time": float(df["timestamp"].iloc[-1]),
            "grad_norm_mean": float(df["grad_norm"].mean()) if "grad_norm" in df else None,
            "grad_norm_max": float(df["grad_norm"].max()) if "grad_norm" in df else None,
        }

    def save(self, filepath: str):
        pd.DataFrame(self.history).to_csv(filepath, index=False)


# ============================================================================
# METRICS CALLBACK
# ============================================================================

class MetricsCallback(TrainerCallback):
    """Custom callback for detailed metrics logging"""

    def __init__(self, tracker: TrainingDynamicsTracker, use_wandb: bool = False):
        self.tracker = tracker
        self.use_wandb = use_wandb and WANDB_AVAILABLE

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        epoch = state.epoch if state.epoch is not None else 0.0
        step = state.global_step

        if "loss" in logs:
            self.tracker.log_step(
                epoch=epoch,
                step=step,
                loss=float(logs["loss"]),
                learning_rate=float(logs.get("learning_rate", 0.0)),
                grad_norm=float(logs.get("grad_norm")) if logs.get("grad_norm") is not None else None,
            )

        if self.use_wandb:
            wandb.log(logs, step=step)


# ============================================================================
# MAIN RESEARCH PIPELINE
# ============================================================================

class CulturalQAResearch:
    """Research pipeline with PEFT QLoRA training + evaluation"""

    def __init__(self, config: ExperimentConfig, use_wandb: bool = False):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.tracker = TrainingDynamicsTracker()

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        logger.info("🚀 Initializing Research Pipeline")
        logger.info(f"   Experiment: {config.experiment_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
        logger.info(f"   Quant: {'4-bit QLoRA' if config.use_4bit else 'No quant'}")

        if self.use_wandb:
            wandb.init(
                project="cultural-qa-research",
                name=config.experiment_name,
                config=asdict(config),
            )

        self.tokenizer = None
        self.model = None

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------

    def _get_model_device(self) -> torch.device:
        # Works for PeftModel and sharded models too (single GPU assumed here)
        return next(self.model.parameters()).device

    def _get_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        if not self.config.use_4bit:
            return None

        # A100 supports bf16 very well
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.bnb_double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================

    def parse_choices(self, choices_str: str) -> Dict[str, str]:
        try:
            return json.loads(choices_str)
        except Exception:
            return {}

    def parse_annotations(self, annotations_str: str) -> List[str]:
        try:
            annotations = json.loads(annotations_str)

            if isinstance(annotations, list):
                all_answers = []
                for item in annotations:
                    if isinstance(item, dict):
                        for key in ["answers", "en_answers", "answer"]:
                            if key in item and item[key]:
                                answers = item[key]
                                if isinstance(answers, list):
                                    all_answers.extend([str(x) for x in answers])
                                else:
                                    all_answers.append(str(answers))
                                break
                return all_answers if all_answers else []

            if isinstance(annotations, dict):
                for key in ["answer", "answers", "en_answers"]:
                    if key in annotations:
                        ans = annotations[key]
                        return ans if isinstance(ans, list) else [str(ans)]
                for v in annotations.values():
                    if isinstance(v, list) and len(v) > 0:
                        return [str(x) for x in v]

            return []
        except Exception:
            return []

    def create_mcq_prompt(self, prompt: str, choices: Dict[str, str], answer: str = None) -> str:
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in cultural knowledge across different regions. Answer the following multiple choice question by selecting the most appropriate option based on cultural context.<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {prompt}

Options:
A. {choices.get('A', '')}
B. {choices.get('B', '')}
C. {choices.get('C', '')}
D. {choices.get('D', '')}

Provide only the letter (A, B, C, or D) of the correct answer.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        if answer:
            text += f"{answer}<|eot_id|>"
        return text

    def create_saq_prompt(self, question: str, answer: str = None) -> str:
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in cultural knowledge across different regions. Answer the following question concisely and accurately based on cultural context.<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}

Provide a brief, accurate answer (1-3 words).<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        if answer:
            text += f"{answer}<|eot_id|>"
        return text

    def prepare_mcq_dataset(self, df: pd.DataFrame) -> Dataset:
        training_texts = []

        for _, row in df.iterrows():
            choices = self.parse_choices(row.get("choices", ""))
            if not choices:
                continue
            prompt_text = self.create_mcq_prompt(
                row["prompt"],
                choices,
                row.get("answer_idx", None)
            )
            training_texts.append(prompt_text)

        logger.info(f"   Prepared {len(training_texts)} MCQ examples")

        dataset = Dataset.from_dict({"text": training_texts})

        def tokenize(examples):
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result

        return dataset.map(tokenize, batched=True, remove_columns=["text"])

    def prepare_saq_dataset(self, df: pd.DataFrame) -> Optional[Dataset]:
        training_texts = []

        logger.info(f"   Processing {len(df)} SAQ rows...")

        for _, row in df.iterrows():
            answers = self.parse_annotations(row.get("annotations", ""))
            if not answers:
                continue
            answer = str(answers[0])
            prompt_text = self.create_saq_prompt(row["question"], answer)
            training_texts.append(prompt_text)

        logger.info(f"   Prepared {len(training_texts)} SAQ examples")

        if len(training_texts) == 0:
            logger.warning("⚠️  No SAQ examples! Skipping SAQ training.")
            return None

        dataset = Dataset.from_dict({"text": training_texts})

        def tokenize(examples):
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result

        return dataset.map(tokenize, batched=True, remove_columns=["text"])

    # =========================================================================
    # MODEL LOADING (FIXED)
    # =========================================================================

    def load_model_for_training(self):
        """Load quantized model + attach PEFT adapters (Trainer-compatible)."""
        logger.info("📥 Loading model for training (PEFT QLoRA)...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        bnb_config = self._get_bnb_config()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
        )

        # Prepare for k-bit training (enables gradients safely)
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

        logger.info("✓ Model loaded with PEFT adapters")

    def load_model_for_inference(self, adapter_dir: Optional[str] = None):
        """Load base model + optionally load PEFT adapter weights."""
        logger.info("📥 Loading model for inference...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        bnb_config = self._get_bnb_config()

        base = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
        )

        if adapter_dir and Path(adapter_dir).exists():
            self.model = PeftModel.from_pretrained(base, adapter_dir)
            logger.info(f"✓ Loaded PEFT adapter from {adapter_dir}")
        else:
            self.model = base
            logger.info("✓ Loaded base model (no adapter)")

        self.model.eval()

    # =========================================================================
    # TRAINING
    # =========================================================================

    def train_model(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            bf16=True,  # A100 best
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset is not None else "no",
            load_best_model_at_end=True if eval_dataset is not None else False,
            report_to="none",
            seed=self.config.seed,
            remove_unused_columns=False,
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
            data_collator=data_collator,
            callbacks=[MetricsCallback(self.tracker, self.use_wandb)],
        )

        trainer.train()

        # Save PEFT adapter + tokenizer + config + training history
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        config_path = f"{self.config.output_dir}/config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        self.tracker.save(f"{self.config.output_dir}/training_history.csv")

    # =========================================================================
    # INFERENCE
    # =========================================================================

    def generate(self, prompt: str, max_new_tokens: int = 50,
                 temperature: float = 0.7, num_samples: int = 1) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt")

        dev = self._get_model_device()
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                num_return_sequences=num_samples,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        responses = []
        for output in outputs:
            full_text = self.tokenizer.decode(output, skip_special_tokens=False)
            if "<|start_header_id|>assistant<|end_header_id|>" in full_text:
                response = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                response = response.split("<|eot_id|>")[0].strip()
                responses.append(response)
            else:
                responses.append(full_text.strip())

        return responses

    def extract_mcq_answer(self, response: str) -> str:
        match = re.search(r"\b([A-D])\b", response.upper())
        if match:
            return match.group(1)

        for char in response.upper():
            if char in ["A", "B", "C", "D"]:
                return char

        return "A"

    def clean_saq_answer(self, response: str) -> str:
        answer = response.strip().lower().split("\n")[0]
        for prefix in ["answer:", "the answer is:", "the answer is", "it is"]:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        return answer.rstrip(".,!?;:")

    def predict_mcq(self, test_file: str, output_file: str) -> Dict:
        logger.info(f"📊 Generating MCQ predictions from {test_file}")

        df = pd.read_csv(test_file)
        results = []
        detailed_results = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="MCQ Prediction"):
            choices = self.parse_choices(row.get("choices", ""))
            if not choices:
                answer = "A"
                confidence = 0.25
            else:
                prompt = self.create_mcq_prompt(row["prompt"], choices)

                if self.config.use_self_consistency:
                    responses = self.generate(
                        prompt,
                        max_new_tokens=10,
                        temperature=self.config.temperature,
                        num_samples=self.config.num_samples,
                    )
                    answers = [self.extract_mcq_answer(r) for r in responses]
                    counts = Counter(answers)
                    answer = counts.most_common(1)[0][0]
                    confidence = counts[answer] / self.config.num_samples
                else:
                    response = self.generate(prompt, max_new_tokens=10, temperature=0.0, num_samples=1)[0]
                    answer = self.extract_mcq_answer(response)
                    confidence = 1.0

            results.append({
                "MCQID": row["MCQID"],
                "A": answer == "A",
                "B": answer == "B",
                "C": answer == "C",
                "D": answer == "D",
            })

            detailed_results.append({
                "MCQID": row["MCQID"],
                "predicted": answer,
                "confidence": float(confidence),
                "prompt": row["prompt"],
            })

        pd.DataFrame(results).to_csv(output_file, sep="\t", index=False)
        pd.DataFrame(detailed_results).to_csv(output_file.replace(".tsv", "_detailed.csv"), index=False)

        logger.info(f"✓ Saved to {output_file}")

        return {
            "num_predictions": len(results),
            "avg_confidence": float(np.mean([r["confidence"] for r in detailed_results])) if detailed_results else 0.0,
        }

    def predict_saq(self, test_file: str, output_file: str) -> Dict:
        logger.info(f"📊 Generating SAQ predictions from {test_file}")

        df = pd.read_csv(test_file)
        results = []
        detailed_results = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="SAQ Prediction"):
            prompt = self.create_saq_prompt(row["question"])

            if self.config.use_self_consistency:
                responses = self.generate(
                    prompt,
                    max_new_tokens=30,
                    temperature=self.config.temperature,
                    num_samples=self.config.num_samples,
                )
                answers = [self.clean_saq_answer(r) for r in responses]
                counts = Counter(answers)
                answer = counts.most_common(1)[0][0]
                confidence = counts[answer] / self.config.num_samples
            else:
                response = self.generate(prompt, max_new_tokens=30, temperature=0.0, num_samples=1)[0]
                answer = self.clean_saq_answer(response)
                confidence = 1.0

            results.append({"ID": row["ID"], "answer": answer})
            detailed_results.append({
                "ID": row["ID"],
                "predicted": answer,
                "confidence": float(confidence),
                "question": row["question"],
            })

        pd.DataFrame(results).to_csv(output_file, sep="\t", index=False)
        pd.DataFrame(detailed_results).to_csv(output_file.replace(".tsv", "_detailed.csv"), index=False)

        logger.info(f"✓ Saved to {output_file}")

        return {
            "num_predictions": len(results),
            "avg_confidence": float(np.mean([r["confidence"] for r in detailed_results])) if detailed_results else 0.0,
            "avg_answer_length": float(np.mean([len(r["predicted"].split()) for r in detailed_results])) if detailed_results else 0.0,
        }

    # =========================================================================
    # COMPLETE EXPERIMENT RUN
    # =========================================================================

    def run_complete_experiment(
        self,
        train_mcq_file: str,
        train_saq_file: str,
        test_mcq_file: str,
        test_saq_file: str,
    ) -> Dict:

        experiment_results = {
            "config": asdict(self.config),
            "training": {},
            "inference": {},
        }

        # TRAINING
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING PHASE")
        logger.info("=" * 70)

        self.load_model_for_training()

        # MCQ training
        logger.info(f"\n📚 Loading MCQ data from {train_mcq_file}")
        mcq_df = pd.read_csv(train_mcq_file)
        train_mcq, val_mcq = train_test_split(mcq_df, test_size=0.1, random_state=self.config.seed)
        logger.info(f"   Split: {len(train_mcq)} train, {len(val_mcq)} val")

        train_ds = self.prepare_mcq_dataset(train_mcq)
        val_ds = self.prepare_mcq_dataset(val_mcq)

        logger.info("\n🚀 Training (MCQ)...")
        self.train_model(train_ds, val_ds)

        experiment_results["training"]["mcq"] = {
            "train_size": len(train_mcq),
            "val_size": len(val_mcq),
            "dynamics": self.tracker.get_summary(),
        }

        # SAQ training (continues same adapter — if you want separate adapters, use separate output_dir)
        logger.info(f"\n📚 Loading SAQ data from {train_saq_file}")
        saq_df = pd.read_csv(train_saq_file)
        train_saq, val_saq = train_test_split(saq_df, test_size=0.1, random_state=self.config.seed)

        saq_train_ds = self.prepare_saq_dataset(train_saq)
        if saq_train_ds is not None:
            saq_val_ds = self.prepare_saq_dataset(val_saq)
            logger.info("\n🚀 Training (SAQ) continuing same adapter...")
            self.train_model(saq_train_ds, saq_val_ds)

            experiment_results["training"]["saq"] = {
                "train_size": len(train_saq),
                "val_size": len(val_saq),
            }

        # INFERENCE
        logger.info("\n" + "=" * 70)
        logger.info("INFERENCE PHASE")
        logger.info("=" * 70)

        adapter_dir = self.config.output_dir
        self.load_model_for_inference(adapter_dir)

        mcq_output = f"{self.config.output_dir}/mcq_prediction.tsv"
        mcq_metrics = self.predict_mcq(test_mcq_file, mcq_output)
        experiment_results["inference"]["mcq"] = mcq_metrics

        saq_output = f"{self.config.output_dir}/saq_prediction.tsv"
        saq_metrics = self.predict_saq(test_saq_file, saq_output)
        experiment_results["inference"]["saq"] = saq_metrics

        logger.info("\n📦 Creating submission.zip...")
        submission_path = f"{self.config.output_dir}/submission.zip"
        with zipfile.ZipFile(submission_path, "w") as zipf:
            zipf.write(mcq_output, "mcq_prediction.tsv")
            zipf.write(saq_output, "saq_prediction.tsv")

        results_path = f"{self.config.output_dir}/experiment_results.json"
        with open(results_path, "w") as f:
            json.dump(experiment_results, f, indent=2)

        if self.use_wandb:
            wandb.log(experiment_results)
            wandb.finish()

        logger.info("\n✅ EXPERIMENT COMPLETE!")
        return experiment_results


# ============================================================================
# ABLATION STUDY RUNNER
# ============================================================================

def run_ablation_study(
    train_mcq: str,
    train_saq: str,
    test_mcq: str,
    test_saq: str,
    output_dir: str = "./ablation_results",
    use_wandb: bool = False,
):
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION STUDY")
    logger.info("=" * 70)

    experiments = []

    # Rank ablation
    for rank in [4, 8, 16, 32, 64]:
        experiments.append(ExperimentConfig(
            lora_rank=rank,
            lora_alpha=rank * 2,
            experiment_name=f"rank_{rank}",
            output_dir=output_dir,
        ))

    # Alpha ratio ablation (rank=16)
    for alpha_ratio in [1, 2, 4, 8]:
        rank = 16
        experiments.append(ExperimentConfig(
            lora_rank=rank,
            lora_alpha=rank * alpha_ratio,
            experiment_name=f"rank16_alpha{rank * alpha_ratio}",
            output_dir=output_dir,
        ))

    # Dropout ablation
    for dropout in [0.0, 0.05, 0.1, 0.2]:
        experiments.append(ExperimentConfig(
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=dropout,
            experiment_name=f"rank16_dropout{dropout}",
            output_dir=output_dir,
        ))

    # Target modules ablation
    target_module_configs = [
        (["q_proj", "v_proj"], "qv_only"),
        (["q_proj", "k_proj", "v_proj", "o_proj"], "qkvo"),
        (["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], "full"),
    ]
    for modules, name in target_module_configs:
        experiments.append(ExperimentConfig(
            lora_rank=16,
            lora_alpha=32,
            target_modules=modules,
            experiment_name=f"modules_{name}",
            output_dir=output_dir,
        ))

    # Learning rate ablation
    for lr in [1e-4, 2e-4, 5e-4, 1e-3]:
        experiments.append(ExperimentConfig(
            lora_rank=16,
            lora_alpha=32,
            learning_rate=lr,
            experiment_name=f"rank16_lr{lr}",
            output_dir=output_dir,
        ))

    all_results = []
    for i, cfg in enumerate(experiments):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EXPERIMENT {i + 1}/{len(experiments)}: {cfg.experiment_name}")
        logger.info(f"{'=' * 70}")

        try:
            pipeline = CulturalQAResearch(cfg, use_wandb=use_wandb)
            results = pipeline.run_complete_experiment(
                train_mcq, train_saq, test_mcq, test_saq
            )
            all_results.append(results)
        except Exception as e:
            logger.error(f"❌ Experiment {cfg.experiment_name} failed: {e}")
            continue

    summary_path = f"{output_dir}/ablation_summary.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n✅ Ablation study complete! Results saved to {summary_path}")
    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cultural QA Research Pipeline")

    parser.add_argument("--mode", choices=["single", "ablation"], default="single",
                        help="Run single experiment or full ablation study")

    # Data files
    parser.add_argument("--train_mcq", default="train_dataset_mcq.csv")
    parser.add_argument("--train_saq", default="train_dataset_saq.csv")
    parser.add_argument("--test_mcq", default="test_dataset_mcq.csv")
    parser.add_argument("--test_saq", default="test_dataset_saq.csv")

    # Single experiment config
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--experiment_name", default="baseline")
    parser.add_argument("--output_dir", default="./experiments")

    # Options
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mode == "ablation":
        run_ablation_study(
            args.train_mcq,
            args.train_saq,
            args.test_mcq,
            args.test_saq,
            args.output_dir,
            args.use_wandb,
        )
    else:
        config = ExperimentConfig(
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            experiment_name=args.experiment_name,
            output_dir=args.output_dir,
            seed=args.seed,
        )

        pipeline = CulturalQAResearch(config, use_wandb=args.use_wandb)
        pipeline.run_complete_experiment(
            args.train_mcq,
            args.train_saq,
            args.test_mcq,
            args.test_saq,
        )


if __name__ == "__main__":
    main()