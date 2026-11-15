# slm-lab — Small Language Model Training, Alignment & Hallucination Lab

An open-source, from-scratch PyTorch Small Language Model (SLM) training pipeline paired with a 6-tab **Visual Learning & Hallucination Lab UI**.

Built to run locally on a single consumer GPU (tested on **NVIDIA RTX 4060, 8 GB VRAM**) and teach every fundamental stage of the modern LLM lifecycle—from pretraining and LoRA to DPO abstention, RAG, and automated evals—with plain-English **`ⓘ` info tooltips** on every setting and metric.

---

## Features

- **100% From-Scratch GPT Architecture (`02_model.py`):** Token + Positional Embeddings, Multi-Head Causal Self-Attention (`scaled_dot_product_attention`), GELU MLP, LayerNorm, Residual Connections, and Weight Tying.
- **Full Training & Alignment Lifecycle:**
  - **Pretraining (`03_train.py`):** Next-token prediction, AdamW, Cosine LR with linear warmup, `bfloat16` mixed precision, gradient accumulation, and validation/overfitting monitoring.
  - **Supervised Fine-Tuning (`05_sft_finetune.py`):** Instruction/Response formatting with prompt loss masking (`ignore_index=-1`).
  - **Abstention SFT + DPO (`07_abstention_sft_and_dpo.py`):** From-scratch Direct Preference Optimization teaching the model when to say *"I don't know based on the provided context."*
  - **LoRA Fine-Tuning (`10_lora_finetune.py`):** From-scratch Low-Rank Adaptation (`W + (alpha/r) * B @ A`) training `<1%` of parameters and merging back for zero-overhead inference.
- **58 Ready-to-Use Domain Datasets:**
  - **Pre-packaged in `data/`:** Finance (`virattt/financial-qa-10K`), Healthcare (`qiaojin/PubMedQA`), Legal (`coastalcph/lex_glue/ledgar`), Science (`allenai/sciq`), and `TinyShakespeare`.
  - **54-Domain Industry Catalog (`domain_catalog.py`):** 1-click download, formatting, tokenization, eval generation, and RAG indexing across Finance, Healthcare, Legal, Code/Cybersecurity, STEM, Enterprise Ops, Hallucination Benchmarks, and Business/ESG.
- **6-Tab Visual Learning UI (`ui_server.py`, `static/index.html`):**
  1. **Data & Tokenizer:** Inspect datasets, download any of the 54 catalog domains in 1 click, and test BPE tokenization live.
  2. **Train Model (Live GPU):** Configure architecture with a live **Parameter & VRAM Estimator** and watch real-time **Train vs. Validation Loss**, **Perplexity**, **Tokens/sec**, and **Overfitting Alerts**.
  3. **Test & Token Confidence:** Autoregressive generation with a **Token Confidence Heatmap** (green `>50%`, yellow `20–50%`, red `<20%`) and hoverable top-3 runner-up probabilities.
  4. **Hallucination Lab:** Compare **Closed-Book (memory only)** vs. **Grounded / RAG-style** vs. **Mismatched Context (Abstention)** side-by-side.
  5. **Automated Evals:** Benchmark any two checkpoints side-by-side on Grounded Accuracy, Abstention Accuracy, Hallucination Rate, and Confidence Calibration.
  6. **Full RAG Pipeline:** Search indexed domain passages via TF-IDF cosine similarity (`09_rag_retriever.py`), inspect Top-K retrieval scores, and feed the `#1` passage into your SLM.

---

## Quickstart

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/abhinavm-web/slm-lab.git
cd slm-lab
python -m venv .venv
```

Activate the environment and install dependencies:
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Install PyTorch with CUDA (adjust cu128/cu124 for your GPU, or use default CPU/MPS)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

```bash
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Visual Lab UI
```bash
python ui_server.py
```
Open **`http://127.0.0.1:8000`** in your browser:
1. **Tab 1:** Pick a pre-packaged domain (`finance`, `healthcare`, `legal`, `sciq`) or download any of the **54 industry domains**.
2. **Tab 2:** Set `Training Steps` (e.g., `600`) and click **Start GPU Training**.
3. **Tabs 3–6:** Inspect token confidence, test Closed-Book vs. Grounded vs. Mismatched Context, run **Automated Evals**, and query the **Full RAG Pipeline**.

---

## CLI Pipeline (Step-by-Step)

```bash
# 1. Prepare TinyShakespeare or any of the 54+ Hugging Face domains
python 01_prepare_data.py
python 06_prepare_hf_dataset.py --domain finance   # or healthcare, legal, sciq, code_sql, etc.

# 2. Inspect model parameter breakdown & static VRAM estimate
python 02_model.py

# 3. Pretrain on GPU
python 03_train.py --max-iters 600

# 4. Generate text with temperature / top-k / top-p sampling
python 04_generate.py --prompt "FIRST CITIZEN:\n" --temperature 0.8

# 5. Supervised Fine-Tuning (SFT) with prompt loss masking
python 05_sft_finetune.py

# 6. Abstention SFT + DPO (teaching "I don't know")
python 07_abstention_sft_and_dpo.py

# 7. Automated Evals & Calibration benchmark
python 08_eval_suite.py --ckpt checkpoints/sciq_abstain_dpo.pt --n 25

# 8. RAG retrieval test
python 09_rag_retriever.py --query "What gas do plants absorb during photosynthesis?"

# 9. LoRA fine-tuning from scratch (trains <1% of weights, then merges)
python 10_lora_finetune.py
```

---

## Repository Structure

| File | Purpose |
|---|---|
| `config.py` | Central hyperparameters, context window, and `bfloat16` precision settings |
| `domain_catalog.py` | Verified catalog of 54+ real-industry Hugging Face domain datasets |
| `01_prepare_data.py` | Downloads TinyShakespeare (or reads custom `.txt`) and tokenizes into `train.bin` / `val.bin` |
| `02_model.py` | From-scratch Decoder-only GPT Transformer + parameter & VRAM estimator |
| `03_train.py` | Pretraining loop with AMP, gradient accumulation, cosine LR, and validation tracking |
| `04_generate.py` | Autoregressive decoding with temperature, top-k, and top-p sampling |
| `05_sft_finetune.py` | Supervised Fine-Tuning (SFT) demonstrating prompt loss masking (`ignore_index=-1`) |
| `06_prepare_hf_dataset.py` | Downloads and formats domain datasets with `<|endoftext|>`, eval samples, and RAG docs |
| `07_abstention_sft_and_dpo.py` | Abstention SFT + from-scratch DPO preference optimization |
| `08_eval_suite.py` | Automated Grounded, Closed-Book, Abstention, and Calibration benchmark |
| `09_rag_retriever.py` | Zero-dependency TF-IDF cosine-similarity retriever over domain passages |
| `10_lora_finetune.py` | From-scratch LoRA (`LoRALinear`) training and weight merging |
| `ui_server.py` | FastAPI backend for live GPU training, confidence heatmaps, evals, and RAG |
| `static/index.html` | 6-tab interactive UI with high-school-level `ⓘ` info tooltips |
| `data/` | Pre-packaged `finance`, `healthcare`, `legal`, `sciq`, and `TinyShakespeare` datasets |

---

## VRAM & Scaling Intuition

For mixed-precision **AdamW training**, static memory is roughly **16 bytes per parameter** (`bf16` weights + `bf16` gradients + `fp32` master copy and Adam moments), plus activations:
- **~30M param SLM** (default config, `gpt2` 50k vocab): ~458 MB static + activations → **~3.2 GB peak VRAM** on an RTX 4060 (8 GB).
- **7B+ param LLM** (full pretraining): >110 GB static before activations → requires multi-GPU clusters or parameter-efficient fine-tuning (**LoRA / QLoRA**).

---

## License

MIT — see [LICENSE](LICENSE).
