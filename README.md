# Cultural Question Answering with LLaMA-3-8B

Parameter-efficient fine-tuning of LLaMA-3-8B for cultural knowledge tasks using LoRA and 4-bit quantization.

## 📊 Results

| Task | Accuracy | Details |
|------|----------|---------|
| **MCQ** | **75%** | China: 89%, UK: 87%, US: 83%, Iran: 61% |
| **SAQ** | **61%** | US: 72%, UK: 65%, Iran: 56%, China: 52% |
| **Overall** | **68%** | +28.5pp over few-shot baseline |

**Codabench Username:** subhan

## 🚀 Quick Start

### Prerequisites

```bash
pip install transformers torch peft datasets pandas tqdm accelerate bitsandbytes
```

### Training & Inference

**MCQ Task:**
```bash
python mcq_finetune_predict.py
```

**SAQ Task:**
```bash
# Full pipeline (train + predict)
python improved_saq_final.py

# Skip training (use existing model)
python improved_saq_final.py --skip_training
```

## 📁 Data Format

**Input Files Required:**
- `train_dataset_mcq.csv` - MCQ training data (836 samples)
- `test_dataset_mcq.csv` - MCQ test data
- `train_dataset_saq.csv` - SAQ training data (1,333 samples)
- `test_dataset_saq.csv` - SAQ test data

**Output Files:**
- `mcq_prediction_finetuned.tsv` - MCQ predictions (format: MCQID, A, B, C, D)
- `saq_prediction_final.tsv` - SAQ predictions (format: ID, answer)

## 🔧 Configuration

### MCQ Configuration
```python
LoRA rank (r): 32
LoRA alpha (α): 32
Learning rate: 2e-4
Epochs: 20
Batch size: 8
Gradient accumulation: 8
Target modules: 7 (q, k, v, o, gate, up, down)
Inference: Greedy decoding (temperature=0)
```

### SAQ Configuration
```python
LoRA rank (r): 32
LoRA alpha (α): 64  # Higher than MCQ
Learning rate: 2e-4
Epochs: 6  # Early stopping
Batch size: 1
Gradient accumulation: 16
Target modules: 7 (q, k, v, o, gate, up, down)
Inference: Self-consistency (11 samples, temperature=0.7)
```

## 🧪 Key Experiments

### LoRA Rank Ablation
| Rank | Params | MCQ | SAQ | Status |
|------|--------|-----|-----|--------|
| 16 | 42M | 0.73 | 0.44 | Underfit |
| **32** | **84M** | **0.75** | **0.61** | **Optimal** |
| 48 | 126M | 0.38 | 0.61 | MCQ overfit |
| 64 | 167M | 0.25 | 0.61 | Severe overfit |

**Key Finding:** Rank 32 provides optimal balance. Higher ranks cause severe MCQ overfitting while SAQ remains stable.

### Hyperparameter Tuning
| LR | Epochs | MCQ | SAQ | Overall |
|----|--------|-----|-----|---------|
| 3e-5 | 4 | 0.68 | 0.52 | 0.60 |
| 2e-4 | 4 | 0.71 | 0.55 | 0.63 |
| **3e-4** | **5** | **0.75** | **0.61** | **0.68** |

**Key Finding:** Higher learning rate (3e-4) with task-specific epochs yields best results.

## 💡 Implementation Highlights

### 1. Task-Specific Data Parsing
- **MCQ:** `json.loads()` for standard JSON format
- **SAQ:** `ast.literal_eval()` for Python dict format
- **Impact:** Correct parser improves accuracy by ~50%

### 2. Self-Consistency for SAQ
```python
# Generate 11 samples with temperature=0.7
outputs = model.generate(
    temperature=0.7,
    num_return_sequences=11,
    do_sample=True
)

# Majority voting
counts = Counter(answers)
most_common = counts.most_common(1)[0]
```
**Impact:** +3-4% accuracy improvement over greedy decoding

### 3. 4-bit Quantization (QLoRA)
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
```
**Benefit:** Reduces memory by ~75%, enables training on single A100 (40GB)

### 4. Special Token Cleaning
Critical post-processing to remove artifacts:
```python
# Remove special tokens and role markers
answer = re.sub(r'<\|[^|]+\|>', '', answer)
answer = re.sub(r'assistant.*', '', answer, flags=re.IGNORECASE)
```

## 📈 Performance by Country

**MCQ Strong Performance:**
- China: 89% (cultural familiarity in training data)
- UK: 87% (Western context)
- US: 83% (Western context)

**MCQ Challenging:**
- Iran: 61% (non-Western, limited training data)

**SAQ Strong Performance:**
- US: 72% (best model alignment)
- UK: 65%

**SAQ Challenging:**
- Iran: 56%
- China: 52%

## 🔬 Technical Details

### Model Architecture
- **Base Model:** LLaMA-3-8B (8 billion parameters)
- **Trainable Parameters:** ~84M (1.03% of total)
- **Quantization:** 4-bit NF4 with double quantization
- **LoRA Modules:** All attention and MLP projections (7 modules)

### Training Infrastructure
- **GPU:** Single NVIDIA A100 (40GB)
- **Training Time:** 
  - MCQ: ~2 hours (20 epochs)
  - SAQ: ~45 minutes (6 epochs)
- **Framework:** PyTorch + HuggingFace Transformers + PEFT

## 📝 Citation

```bibtex
@misc{ali2024cultural,
  title={Parameter-Efficient Adaptation of Large Language Models for Cultural Knowledge Tasks},
  author={Ali, Subhan and Shah, Khadijah Ali},
  year={2024},
  institution={TU Dresden}
}
```

## 🙏 Acknowledgments

- Course: Behind the Secrets of Large Language Models
- Institution: TU Dresden
- Competition: [Codabench Cultural QA Challenge](https://www.codabench.org/competitions/11605/)

## 📧 Contact

- **Subhan Ali:** subhan.ali@tu-dresden.de
- **Khadijah Ali Shah:** khadijah_ali.shah@mailbox.tu-dresden.de

---

**Matriculation Numbers:**
- Subhan Ali: 5192655
- Khadijah Ali Shah: 5195704
