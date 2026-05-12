# Interpretable Emotion Attribution in Social Graphs

[![Conference](https://img.shields.io/badge/MathAI-2026-blue)](https://mathai2026.github.io)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Aigents-2496ED?logo=docker)](https://www.docker.com)
[![Colab](https://img.shields.io/badge/Colab-RoBERTa%20%7C%20Llama--3-F9AB00?logo=googlecolab)](https://colab.research.google.com)

> **Published at MathAI 2026** — Department of Mathematics & Mechanics, Novosibirsk State University

A systematic comparison of three modeling paradigms for directed relational emotion attribution in social graphs:
**Aigents** (rule-based), **RoBERTa-GoEmotions** (fine-tuned Transformer), and **Llama-3-8B** (few-shot LLM).

---

## Key Results

| Model | N | Macro-F1 | 95% CI | Interpretable? |
|---|---|---|---|---|
| RoBERTa-3turn | 150 | **0.636** | [0.554, 0.715] | ✗ |
| Aigents (Mode 3) | 97 | 0.624 | [0.516, 0.727] | ✓ |
| Aigents (Mode 1) | 114 | 0.625 | [0.526, 0.712] | ✓ |
| RoBERTa-1turn | 150 | 0.617 | [0.579, 0.655] | ✗ |
| Aigents (Mode 2) | 150 | 0.582 | [0.500, 0.659] | ✓ |
| RoBERTa-baseline | 150 | 0.538 | [0.496, 0.580] | ✗ |
| Llama-3 (few-shot) | 150 | 0.528 | [0.484, 0.568] | ✗ |

**Main finding:** The interpretable Aigents system achieves statistically equivalent performance to RoBERTa-3turn (overlapping 95% bootstrap CIs), formally refuting the performance–interpretability dilemma for this task. Few-shot Llama-3 significantly underperforms both alternatives.

---

## Repository Structure

```
.
├── Notebooks/
│   ├── 01_relation_extraction_deberta.ipynb   # DeBERTa-v3 RE pipeline
│   ├── 02_silver_label_generation_llama3.ipynb # LLM-assisted annotation
│   ├── 03_roberta_finetuning_1turn.ipynb      # RoBERTa-1turn fine-tuning
│   ├── 04_roberta_finetuning_3turn.ipynb      # RoBERTa-3turn fine-tuning
│   ├── 05_aigents_evaluation.ipynb            # Rule-based evaluation
│   └── 06_llama3_ablation.ipynb               # Few-shot LLM ablation
│
├── data/
│   ├── dialogre_v2/                           # DialogRE v2 dataset
│   ├── gold_standard_150.json                 # 150-sample annotated test set
│   └── silver_labels_1179.json                # LLM-generated training labels
│
├── deberta-v3-large_model/                    # Fine-tuned DeBERTa-v3 checkpoint
│
├── Dockerfile                                 # Aigents Docker environment
├── docker-compose.yml                         # Docker Compose configuration
├── requirements_aigents.txt                   # Dependencies for Aigents (CPU)
├── requirements_colab.txt                     # Dependencies for Colab (GPU)
└── README.md
```

---

## Setup and Installation

The project uses **two separate environments**:
- **Docker** — for the Aigents rule-based system (CPU-only, no GPU required)
- **Google Colab** — for RoBERTa fine-tuning and Llama-3 evaluation (T4 GPU)

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) (for Aigents)
- Google account with Colab access (for neural models)
- Python 3.10+

---

## Running the Models

### 1. Relation Extraction Pipeline (DeBERTa-v3-Large)

This step extracts candidate entity pairs from the DialogRE test split and constructs the 3-turn context windows used for all downstream models.

**Run in Colab:**

```bash
# Open Notebook 01 in Google Colab
# Runtime > Change runtime type > T4 GPU
```

Install dependencies:
```python
!pip install -r requirements_colab.txt
```

The notebook fine-tunes `microsoft/deberta-v3-large` on DialogRE and extracts
2,610 candidate sentiment relations from the test split. The pre-trained checkpoint
is available in the `deberta-v3-large_model/` directory — you can skip training
and run inference directly.

Key hyperparameters:
```python
MAX_SEQ_LEN = 512
BATCH_SIZE  = 2          # with gradient accumulation steps = 4
EPOCHS      = 5
LR          = 1e-5
```

---

### 2. Silver Label Generation (Llama-3-8B-Instruct)

Generates training labels for 1,179 instances from the DialogRE training split
using Llama-3-8B-Instruct with 4-bit NF4 quantization.

**Run in Colab** (`Notebook 02`):

```python
!pip install -r requirements_colab.txt
```

```python
# Key settings
MODEL_ID        = "meta-llama/Meta-Llama-3-8B-Instruct"
QUANTIZATION    = "4-bit NF4"
CONFIDENCE_THRESHOLD = 0.70
N_SHOT          = 10        # Few-shot demonstrations
TEMPERATURE     = 0.1       # Greedy-like decoding
```

> **Note:** Llama-3 access requires a HuggingFace account with gated model access.
> Apply at: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

Output: `data/silver_labels_1179.json` — already provided in this repository
if you wish to skip this step.

---

### 3. RoBERTa Fine-Tuning

Two variants are trained: single-turn (1T) and three-turn context (3T).
The base checkpoint is initialized from a GoEmotions-specialized RoBERTa-large.

**Run in Colab** (`Notebook 03` for 1-turn, `Notebook 04` for 3-turn):

```python
!pip install -r requirements_colab.txt
```

```python
# 1-turn configuration
MODEL_NAME  = "Lakssssshya/RoBERTa-large-goemotions"
MAX_LEN     = 128
LR          = 8e-6
BATCH_SIZE  = 16
EPOCHS      = 15
SEED        = 19            # Best seed from 5-seed ablation

# 3-turn configuration (Notebook 04)
MAX_LEN     = 256
LR          = 9e-6          # Note: differs from 1-turn
```

Both notebooks implement early stopping with patience of 3 validation epochs
based on Macro-F1 score. Training typically converges within 3–5 epochs on a T4 GPU.

**Expected validation results at convergence:**

| Variant | Val Accuracy | Val Macro-F1 | Val F1-Pos | Val F1-Neg | Val F1-Neu |
|---|---|---|---|---|---|
| 1-turn | 0.809 | 0.809 | 0.845 | 0.803 | 0.779 |
| 3-turn | 0.835 | 0.836 | 0.848 | 0.850 | 0.810 |

---

### 4. Aigents Rule-Based Evaluation

Aigents runs entirely on CPU inside a Docker container. No GPU is required.

**Step 1 — Build and start the container:**

```bash
docker-compose up --build
```

**Step 2 — Run evaluation:**

```bash
# Enter the running container
docker exec -it <container_name> bash

# Install Aigents dependencies
pip install -r requirements_aigents.txt

# Run evaluation (all three modes)
python evaluate_aigents.py --mode 1    # Binary, gold neutral excluded (N=114)
python evaluate_aigents.py --mode 2    # Neutral mapped to dominant class (N=150)
python evaluate_aigents.py --mode 3    # Detected neutral excluded (N=97)
```

**Aigents scoring parameters:**

```python
# Neutral detection thresholds (empirically tuned on dev set)
TAU_MARGIN    = 0.2    # Minimum margin |P - |N|| for non-neutral
TAU_RAW       = 0.3    # Minimum raw strength for non-neutral

# Strength normalization
# s(R) = 1 - exp(-1.5 * R)
LAMBDA        = 1.5

# Lexicon sizes
POSITIVE_NGRAMS = 3800+
NEGATIVE_NGRAMS = 8200+
```

**Expected test results:**

| Mode | N | Accuracy | F1-Pos | F1-Neg | Macro-F1 |
|---|---|---|---|---|---|
| Mode 1 | 114 | 0.667 | 0.750 | 0.500 | 0.625 |
| Mode 2 | 150 | 0.593 | 0.651 | 0.512 | 0.582 |
| Mode 3 | 97  | 0.701 | 0.794 | 0.453 | 0.624 |

---

### 5. Llama-3 Few-Shot Ablation Study

Evaluates Llama-3-8B-Instruct as a direct classifier across 9 prompt configurations
(3 templates × 3 shot counts).

**Run in Colab** (`Notebook 06`):

```python
# Ablation grid
TEMPLATES  = ["standard", "chain_of_thought", "json_format"]
SHOT_COUNTS = [3, 5, 10]

# Full results (Macro-F1):
#                3-shot  5-shot  10-shot  Mean
# Standard:      0.501   0.447   0.488    0.479
# Chain-of-Thought: 0.394  0.296  0.294   0.328
# JSON-Format:   0.382   0.356   0.390    0.376
```

---

## Dataset

This project uses [DialogRE v2](https://github.com/nlpdata/dialogre) — the first
human-annotated dialogue-based relation extraction dataset derived from
*Friends* TV show transcripts.

| Split | Dialogues | Relation Instances |
|---|---|---|
| Train | 1,073 | ~8,600 |
| Development | 358 | ~1,100 |
| Test | 357 | 2,610 extracted / 150 annotated |

The `data/gold_standard_150.json` file contains the 150-instance human-annotated
test set constructed for this work:
- 70 positive (46.7%), 44 negative (29.3%), 36 neutral (24.0%)
- Inter-annotator Cohen's Kappa: κ = 0.579 (moderate agreement)

> **License note:** DialogRE is distributed under its own license.
> Please refer to the [original repository](https://github.com/nlpdata/dialogre)
> for terms of use before redistributing the data.

---

## Reproducing the Main Results

To reproduce the primary comparison in Table 5.1 of the thesis from scratch:

```bash
# 1. Extract entity pairs (Notebook 01, ~30 min on T4)
# 2. Generate silver labels (Notebook 02, ~2 hours on T4)
# 3. Fine-tune RoBERTa-3turn (Notebook 04, ~20 min on T4, seed=19)
# 4. Run Aigents Mode 3 (docker-compose + evaluate_aigents.py --mode 3, ~5 min CPU)
# 5. Compare results against gold_standard_150.json
```

To skip directly to evaluation using pre-generated data:

```bash
# Silver labels and test set are already in data/
# DeBERTa checkpoint is already in deberta-v3-large_model/
# Proceed directly to steps 3–5 above
```

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@inproceedings{gidado2026interpretable,
  title     = {Interpretable Emotion Attribution in Social Graphs:
               A Comparative Analysis of Rule-Based, Transformer, and LLM Models},
  author    = {Gidado, Usman and Kolonin, Anton},
  booktitle = {Proceedings of MathAI 2026},
  year      = {2026},
  address   = {Novosibirsk, Russia},
  institution = {Department of Mathematics \& Mechanics,
                 Novosibirsk State University}
}
```

---

## Dependencies

### Aigents (CPU / Docker)
```
pygents
nltk
pandas
scikit-learn
numpy
```
See `requirements_aigents.txt` for pinned versions.

### Neural Models (Colab / GPU)
```
transformers>=4.40.0
torch>=2.1.0
datasets
peft
bitsandbytes       # 4-bit quantization for Llama-3
accelerate
scikit-learn
pandas
numpy
```
See `requirements_colab.txt` for pinned versions.

---

## Related Resources

- **Aigents / Pygents framework:** https://github.com/aigents/pygents/tree/main/pygents
- **RoBERTa-GoEmotions checkpoint:** https://huggingface.co/Lakssssshya/RoBERTa-large-goemotions
- **DeBERTa-v3-large:** https://huggingface.co/microsoft/deberta-v3-large
- **Llama-3-8B-Instruct:** https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- **DialogRE dataset:** https://github.com/nlpdata/dialogre

---

## Authors

**Usman Babayo Gidado** — Department of Mathematics & Mechanics, Novosibirsk State University
`ubgidadoac@gmail.com`

**Anton Kolonin** (Supervisor) — Department of Mathematics & Mechanics, Novosibirsk State University
`akolonin@gmail.com`

---

## License

This project is released under the MIT License. See `LICENSE` for details.
The DialogRE dataset is subject to its own separate license terms.
