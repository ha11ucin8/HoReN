# HoReN: Normalized Hopfield Retrieval for Large-Scale Sequential Model Editing

Authors: *Yuan Fang, Yi Xie, and Xuming Ran*

Experiment codebase for the HoReN paper. Contains implementations of MEND, DEFER, ROME, GRACE, WISE, AlphaEdit, HoReN, UltraEdit, and UnKE run through unified scripts on ZsRE, WikiBigEdit, and UnKE unstructured data.

## Attribution

The `src/` library is adapted from [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit) and trimmed to include only the editors used in this work.

---

## Repository structure

```
horen-paper/
├── src/             # Trimmed EasyEdit library (editors, models, dataset, util)
│   ├── editors/            # BaseEditor and related utilities
│   ├── models/             # mend, defer, rome, grace, wise, alphaedit, horen, ultraedit, unke, ft, ike, kn
│   ├── dataset/            # ZsreDataset, WikiBigEditDataset, AKEWUnifiedDataset
│   ├── evaluate/           # compute_edit_quality, eval_akew_unstructured
│   ├── trainer/            # training utilities (used by MEND)
│   └── util/               # alg_dict, globals, hparams base class, nethook, etc.
├── hparams/                # Per-editor, per-model YAML/JSON configs
│   ├── MEND/               # llama3-8b, llama3.1-8b, qwen2.5-7b, deepseek-r1-distill-qwen-1.5b
│   ├── DEFER/              # same four models
│   ├── ROME/               # same four models
│   ├── GRACE/              # same four models + deepseek-r1-distill-llama-8b, gpt-oss-20b
│   ├── WISE/               # same four models + deepseek-r1-distill-llama-8b, gpt-oss-20b
│   ├── AlphaEdit/          # same four models
│   ├── HOREN/              # all six models
│   ├── ULTRAEDIT/          # llama3-8b, llama3.1-8b, qwen2.5-7b, deepseek-r1-distill-llama-8b, deepseek-r1-distill-qwen-1.5b
│   └── UNKE/               # llama3-8b, llama3.1-8b, qwen2.5-7b, deepseek-r1-distill-llama-8b, deepseek-r1-distill-qwen-1.5b (JSON)
├── data/                   # (not committed) place datasets here
│   ├── ZsRE/
│   │   ├── zsre_edit_data.json
│   │   └── zsre_train_data.json
│   ├── WikiBigEdit/
│   │   └── wiki_big_edit.json
│   ├── UnKE/
│   │   └── final_data_v3.json      # UnKE unstructured edit samples
│   └── alpaca_data.json            # Alpaca instruction data (UNKE locality only)
├── outputs/                # (not committed) experiment result JSON files
├── logs/                   # (not committed) per-checkpoint log files
├── experiments/
│   ├── run_structured_data_editing.py    # Structured editing (ZsRE / WikiBigEdit)
│   └── run_unstructured_data_editing.py  # Unstructured editing (UnKE dataset)
├── requirements.txt
├── pyproject.toml          # black formatter config
└── .gitignore
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download models

Place model weights under `./hugging_cache/`. The path must match the `model_name` field in the YAML config. Example layout:

```
hugging_cache/
├── llama3-8b-instruct/
├── llama3.1-8b-instruct/
├── qwen2.5-7b-instruct/
├── deepseek-r1-distill-qwen-1.5b/
├── deepseek-r1-distill-llama-8b/   # HOREN / GRACE / WISE only
└── gpt-oss-20b/                    # HOREN / GRACE / WISE only
```

### 3. Download datasets

**ZsRE** (question-answering factual edits):

Place `zsre_edit_data.json` and `zsre_train_data.json` in `./data/ZsRE/`.

**WikiBigEdit** (lifelong factual edits from Wikidata):

Download from [HuggingFace](https://huggingface.co/datasets/lukasthede/WikiBigEdit) and place the JSON file at `./data/WikiBigEdit/wiki_big_edit.json`.

### 4. MEND: train a meta-network first

MEND requires a pre-trained meta-network (stored at the `archive` path in the YAML). Train it with EasyEdit's training pipeline before running evaluation. The YAML's `eval_only: True` flag enables inference-only mode.

### 5. AlphaEdit: null-space projection matrix

AlphaEdit computes a null-space projection matrix `P` before editing. If the file at `P_loc` does not exist, it is computed automatically (downloads a Wikipedia corpus and runs covariance estimation — one-time cost of ~20 min). The matrix is saved locally for subsequent runs.

### 6. UltraEdit: no pre-computation required

UltraEdit is training-free and requires no pre-computation. It uses a `lifelong_normalizer` (online running statistics) that accumulates across sequential edits, and a closed-form weight update via linear solve. Run it directly with `--sequential_edit`.

### 7. Unstructured data: download UnKE dataset and Alpaca data

For unstructured editing experiments, place the following files under `./data/`:

- `data/UnKE/final_data_v3.json` — UnKE unstructured edit samples (question + answer + paraphrase + sub-questions).
- `data/alpaca_data.json` — Alpaca instruction-following data, used by UNKE for locality regularisation. Download from [tatsu-lab/alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json). Not required for HOREN.

---

## Running experiments

### Structured data (ZsRE / WikiBigEdit)

Entry point: `experiments/run_structured_data_editing.py`. Run from the repo root with `PYTHONPATH=$(pwd)`:

```bash
PYTHONPATH=$(pwd) python experiments/run_structured_data_editing.py \
    --editing_method <METHOD> \
    --hparams_dir ./hparams/<METHOD>/<model>.yaml \
    --data_dir <data_path> \
    --data_type <ZsRE|WikiBigEdit> \
    --ds_size <N> \
    --output_dir ./outputs \
    [--sequential_edit]
```

#### Arguments

| Argument | Required | Description |
|---|---|---|
| `--editing_method` | yes | One of: `MEND`, `DEFER`, `ROME`, `GRACE`, `WISE`, `AlphaEdit`, `HOREN`, `ULTRAEDIT` |
| `--hparams_dir` | yes | Path to the YAML config file |
| `--data_dir` | yes | Directory containing ZsRE JSONs **or** path to WikiBigEdit JSON file |
| `--data_type` | yes | `ZsRE` or `WikiBigEdit` |
| `--ds_size` | no | Number of edits to run (default: 100) |
| `--output_dir` | no | Directory for result JSON (default: `./outputs`) |
| `--sequential_edit` | no | Apply edits cumulatively on the same model instance |

#### Example commands

```bash
# HOREN on LLaMA-3-8B, ZsRE, 1000 sequential edits
PYTHONPATH=$(pwd) python experiments/run_structured_data_editing.py \
    --editing_method HOREN \
    --hparams_dir ./hparams/HOREN/llama3-8b.yaml \
    --data_dir ./data/ZsRE \
    --data_type ZsRE \
    --ds_size 1000 \
    --output_dir ./outputs \
    --sequential_edit

# WISE on Qwen2.5-7B, ZsRE, 1000 sequential edits
PYTHONPATH=$(pwd) python experiments/run_structured_data_editing.py \
    --editing_method WISE \
    --hparams_dir ./hparams/WISE/qwen2.5-7b.yaml \
    --data_dir ./data/ZsRE \
    --data_type ZsRE \
    --ds_size 1000 \
    --output_dir ./outputs \
    --sequential_edit

# ROME on LLaMA-3.1-8B, ZsRE, 100 edits (non-sequential)
PYTHONPATH=$(pwd) python experiments/run_structured_data_editing.py \
    --editing_method ROME \
    --hparams_dir ./hparams/ROME/llama3.1-8b.yaml \
    --data_dir ./data/ZsRE \
    --data_type ZsRE \
    --ds_size 100 \
    --output_dir ./outputs

# HOREN on DeepSeek-R1-Distill-Qwen-1.5B, WikiBigEdit
PYTHONPATH=$(pwd) python experiments/run_structured_data_editing.py \
    --editing_method HOREN \
    --hparams_dir ./hparams/HOREN/deepseek-r1-distill-qwen-1.5b.yaml \
    --data_dir ./data/WikiBigEdit/wiki_big_edit.json \
    --data_type WikiBigEdit \
    --ds_size 1000 \
    --output_dir ./outputs \
    --sequential_edit

# AlphaEdit on DeepSeek-R1-Distill-Qwen-1.5B, ZsRE, 1000 sequential edits
PYTHONPATH=$(pwd) python experiments/run_structured_data_editing.py \
    --editing_method AlphaEdit \
    --hparams_dir ./hparams/AlphaEdit/deepseek-r1-distill-qwen-1.5b.yaml \
    --data_dir ./data/ZsRE \
    --data_type ZsRE \
    --ds_size 1000 \
    --output_dir ./outputs \
    --sequential_edit

# UltraEdit on LLaMA-3-8B, ZsRE, 1000 sequential edits
PYTHONPATH=$(pwd) python experiments/run_structured_data_editing.py \
    --editing_method ULTRAEDIT \
    --hparams_dir ./hparams/ULTRAEDIT/llama3-8b.yaml \
    --data_dir ./data/ZsRE \
    --data_type ZsRE \
    --ds_size 1000 \
    --output_dir ./outputs \
    --sequential_edit
```

---

### Unstructured data (UnKE dataset)

Entry point: `experiments/run_unstructured_data_editing.py`. Supports **HOREN** and **UNKE**. Edits are always applied sequentially (one-by-one, accumulating on the same model).

```bash
PYTHONPATH=$(pwd) python experiments/run_unstructured_data_editing.py \
    --editing_method <HOREN|UNKE> \
    --hparams_dir ./hparams/<METHOD>/<model>.<yaml|json> \
    --data_dir ./data \
    --data_type unke \
    --ds_size <N> \
    --output_dir ./outputs \
    [--log_dir ./logs]
```

#### Arguments

| Argument | Required | Description |
|---|---|---|
| `--editing_method` | yes | `HOREN` or `UNKE` |
| `--hparams_dir` | yes | `.yaml` for HOREN, `.json` for UNKE |
| `--data_dir` | yes | Parent data directory (must contain `UnKE/final_data_v3.json`; UNKE also needs `alpaca_data.json` at this level) |
| `--data_type` | yes | `unke` (or `counterfact`, `wikiupdate`, `mquake` if available) |
| `--ds_size` | no | Number of edits to run (default: 100) |
| `--output_dir` | no | Directory for final result JSON (default: `./outputs`) |
| `--log_dir` | no | Directory for checkpoint files (default: `./logs`) |
| `--batch_size` | no | Edits per batch (default: 1) |
| `--bert_model_path` | no | SentenceTransformer model for BERTScore (default: `sentence-transformers/all-MiniLM-L6-v2`) |
| `--device` | no | CUDA device index (default: 0) |
| `--seed` | no | Random seed (default: 2024) |

#### Example commands

```bash
# HOREN on LLaMA-3-8B, UnKE, 1000 sequential edits
PYTHONPATH=$(pwd) python experiments/run_unstructured_data_editing.py \
    --editing_method HOREN \
    --hparams_dir ./hparams/HOREN/llama3-8b.yaml \
    --data_dir ./data \
    --data_type unke \
    --ds_size 1000 \
    --output_dir ./outputs

# UNKE on LLaMA-3-8B, UnKE, 1000 sequential edits
PYTHONPATH=$(pwd) python experiments/run_unstructured_data_editing.py \
    --editing_method UNKE \
    --hparams_dir ./hparams/UNKE/llama3-8b.json \
    --data_dir ./data \
    --data_type unke \
    --ds_size 1000 \
    --output_dir ./outputs

# UNKE on Qwen2.5-7B, UnKE, 1000 sequential edits
PYTHONPATH=$(pwd) python experiments/run_unstructured_data_editing.py \
    --editing_method UNKE \
    --hparams_dir ./hparams/UNKE/qwen2.5-7b.json \
    --data_dir ./data \
    --data_type unke \
    --ds_size 1000 \
    --output_dir ./outputs
```

---

## Output format

### Structured editing

Each run writes two outputs:

1. **`outputs/<data_type>_<model>_<method>_N=<N>_Sequential=<bool>_<timestamp>.json`** — full per-edit metrics list.
2. **`logs/<prefix>_Checkpoint_<N>_results.json`** — aggregated summary + per-edit metrics at each checkpoint (written by `summary_metrics_` inside `BaseEditor`).

Checkpoints fire at edit counts: `1, 10, 30, 100, 120, 500` and every `1000` steps (for runs ≤ 1000 edits).

#### Structured metrics

| Metric | Field | Description |
|---|---|---|
| Reliability | `rewrite_acc` | Whether the model now produces the target answer |
| Generalization | `rephrase_acc` | Whether the edit holds for paraphrased prompts |
| Locality | `locality.neighborhood_acc` | Whether unrelated knowledge is preserved |
| Portability (WikiBigEdit) | `portability.personas_acc`, `portability.mhop_acc` | Generalization to personas / multi-hop chains |

### Unstructured editing

Each run writes:

1. **`outputs/unstructured_<model>_<method>_<data_type>_N=<N>_<timestamp>.json`** — combined `{"metrics": {...}, "samples": [...]}` for the full run.
2. **`logs/unstructured_<model>_<method>_<data_type>_N=<N>_<timestamp>_Checkpoint_<N>_results.json`** — same combined format at each checkpoint.

Checkpoints follow the same schedule as structured editing.

#### Unstructured metrics

| Metric | Field | Description |
|---|---|---|
| Reliability | `Original.BLEU SCORE`, `Original.ROUGE-{1,2,L}`, `Original.Bert Score` | Scores on the original (direct) question |
| Generalization | `Para.BLEU SCORE`, `Para.ROUGE-{1,2,L}`, `Para.Bert Score` | Scores on the paraphrased question |
| Portability | `Sub.ROUGE-{1,2,L}` | Scores on sub-questions (multi-hop reasoning) |

---

## Code formatting

```bash
black .
```

Black is configured in `pyproject.toml` (line length 120, excludes `.venv`, `venv`).

---

## Hyperparameters

Each `hparams/<METHOD>/<model>.yaml` (or `.json` for UNKE) specifies the hyper parameters of the corresponding editing method and experiment.

---

## Citing

If you use this code, please cite:

```bibtex
@misc{horen2026,
  title  = {HoReN: Normalized Hopfield Retrieval for Large-Scale Sequential Model Editing},
  author = {Fang, Yuan and Xie, Yi and Ran, Xuming},
  year   = {2026},
  note   = {Code: \url{https://github.com/ha11ucin8/HoReN}}
}
```

This codebase builds on EasyEdit and the baselines below. Please also cite the relevant works:

```bibtex
@article{wang2024easyedit,
  title   = {EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models},
  author  = {Wang, Peng and Zhang, Ningyu and Tian, Bozhong and Xi, Zekun and Yao, Yunzhi and Xu, Ziwen and Wang, Mengru and Mao, Shengyu and Wang, Xiaohan and Cheng, Siyuan and Liu, Kangwei and Ni, Yuansheng and Zheng, Guozhou and Chen, Huajun},
  journal = {arXiv preprint arXiv:2308.07269},
  year    = {2023}
}

@inproceedings{meng2022locating,
  title     = {Locating and Editing Factual Associations in {GPT}},
  author    = {Meng, Kevin and Bau, David and Andonian, Alex and Belinkov, Yonatan},
  booktitle = {NeurIPS},
  year      = {2022}
}

@inproceedings{meng2023memit,
  title     = {Mass-Editing Memory in a Transformer},
  author    = {Meng, Kevin and Sharma, Arnab Sen and Andonian, Alex and Belinkov, Yonatan and Bau, David},
  booktitle = {ICLR},
  year      = {2023}
}

@inproceedings{mitchell2022mend,
  title     = {Fast Model Editing at Scale},
  author    = {Mitchell, Eric and Lin, Charles and Bosselut, Antoine and Finn, Chelsea and Manning, Christopher D.},
  booktitle = {ICLR},
  year      = {2022}
}

@inproceedings{hartvigsen2023grace,
  title     = {Aging with {GRACE}: Lifelong Model Editing with Discrete Key-Value Adaptors},
  author    = {Hartvigsen, Thomas and Sankaranarayanan, Swami and Palangi, Hamid and Kim, Yoon and Ghassemi, Marzyeh},
  booktitle = {NeurIPS},
  year      = {2023}
}

@inproceedings{wang2024wise,
  title     = {{WISE}: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models},
  author    = {Wang, Peng and Li, Zexi and Zhang, Ningyu and Xu, Ziwen and Yao, Yunzhi and Jiang, Yong and Xie, Pengjun and Huang, Fei and Chen, Huajun},
  booktitle = {NeurIPS},
  year      = {2024}
}

@inproceedings{fang2025alphaedit,
  title     = {{AlphaEdit}: Null-Space Constrained Knowledge Editing for Language Models},
  author    = {Fang, Junfeng and Jiang, Houcheng and Wang, Kun and Ma, Yunshan and Wang, Xiang and He, Xiangnan and Chua, Tat-Seng},
  booktitle = {ICLR},
  year      = {2025}
}

@article{deng2024unke,
  title   = {{UnKE}: Unstructured Knowledge Editing in Large Language Models},
  author  = {Deng, Jingcheng and Wei, Zihao and Pang, Liang and Ding, Hanxing and Shen, Huawei and Cheng, Xueqi},
  journal = {arXiv preprint arXiv:2405.15349},
  year    = {2024}
}

@inproceedings{levy2017zsre,
  title     = {Zero-Shot Relation Extraction via Reading Comprehension},
  author    = {Levy, Omer and Seo, Minjoon and Choi, Eunsol and Zettlemoyer, Luke},
  booktitle = {CoNLL},
  year     = {2017}
}

@misc{thede2024wikibigedit,
  title  = {{WikiBigEdit}: A Lifelong Knowledge Editing Benchmark from Wikidata},
  author = {Thede, Lukas and others},
  year   = {2024},
  url    = {https://huggingface.co/datasets/lukasthede/WikiBigEdit}
}
```
