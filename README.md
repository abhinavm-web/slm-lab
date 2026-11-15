# slm-lab — Small Language Model Training & Hallucination Lab

An open-source, from-scratch PyTorch Small Language Model (SLM) training pipeline paired with an interactive **Visual Learning & Hallucination Lab UI**.

Designed to run locally on a single consumer GPU (tested on **NVIDIA RTX 4060, 8 GB VRAM**) and teach every fundamental stage of training, evaluating, and debugging language models—with plain-English **`ⓘ` info tooltips** on every setting and metric.

---

## Features

- **100% From-Scratch GPT Architecture (`02_model.py`):** Token + Positional Embeddings, Multi-Head Causal Self-Attention (`scaled_dot_product_attention`), GELU MLP, LayerNorm, Residual Connections, and Weight Tying.
- **Pretraining & Post-Training (SFT) (`03_train.py`, `05_sft_finetune.py`):** Next-token prediction, AdamW, Cosine LR with linear warmup, `bfloat16` mixed precision, gradient accumulation, validation/overfitting monitoring, and prompt-masked Supervised Fine-Tuning.
- **Interactive Visual UI (`ui_server.py`, `static/index.html`):**
  1. **Data & Tokenizer Playground:** Inspect prepared datasets and see how BPE subword tokenization splits any sentence into token IDs.
  2. **Live GPU Training:** Adjust model architecture & hyperparameters with a live **Parameter & VRAM Estimator**, and watch real-time **Train vs. Validation Loss**, **Perplexity**, **Tokens/sec**, and **Overfitting Alerts**.
  3. **Test & Token Confidence:** Autoregressive generation with a **Token Confidence Heatmap** (green `>50%`, yellow `20–50%`, red `<20%`) and hoverable top-3 runner-up token probabilities.
  4. **Hallucination Lab:** Compare **Closed-Book (memory only)** vs. **Grounded / RAG-style (with textbook context)** side-by-side on unseen validation questions.
- **Pre-Packaged Datasets (`data/`):** Includes both **Science Q&A (`allenai/sciq`)** (~1.45M train tokens, 100 held-out eval samples) and **TinyShakespeare** (~304k train tokens) ready to train out of the box.

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
# Install PyTorch with CUDA (adjust cu128/cu124 for your GPU, or use default CPU/MPS on Mac)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

```bash
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Visual Learning & Hallucination Lab UI
```bash
python ui_server.py
```
Open **`http://127.0.0.1:8000`** in your browser:
1. Explore **Tab 1 (Data & Tokenizer)** to see how `allenai/sciq` is tokenized.
2. Go to **Tab 2 (Train Model)**, set `Training Steps` (e.g., `600`), and click **Start GPU Training**.
3. Test your trained checkpoint in **Tab 3 (Test & Token Confidence)** and **Tab 4 (Hallucination Lab)**.

---

## CLI Pipeline (Step-by-Step)

You can also run every stage directly from the terminal:

```bash
# Stage 1: Prepare TinyShakespeare (or pass --input path/to/your.txt)
python 01_prepare_data.py

# Stage 1b: Re-prepare Science Q&A (allenai/sciq) from Hugging Face
python 06_prepare_hf_dataset.py

# Stage 2: Inspect model parameter breakdown & static VRAM estimate
python 02_model.py

# Stage 3: Pretrain on GPU
python 03_train.py --max-iters 600

# Stage 4: Generate text with temperature / top-k / top-p sampling
python 04_generate.py --prompt "FIRST CITIZEN:\n" --temperature 0.8

# Stage 5: Supervised Fine-Tuning (SFT) with prompt loss masking
python 05_sft_finetune.py
python 04_generate.py --ckpt checkpoints/sft_ckpt.pt --prompt "### Instruction:\nSummarize the duty of a wise ruler.\n\n### Response:\n"
```

---

## Repository Structure

| File | Purpose |
|---|---|
| `config.py` | Central hyperparameters, context window, and `bfloat16` precision settings |
| `01_prepare_data.py` | Downloads TinyShakespeare (or reads custom `.txt`) and tokenizes into `train.bin` / `val.bin` |
| `02_model.py` | From-scratch Decoder-only GPT Transformer + parameter & VRAM estimator |
| `03_train.py` | Pretraining loop with AMP, gradient accumulation, cosine LR, and validation tracking |
| `04_generate.py` | Autoregressive decoding with temperature, top-k, and top-p sampling |
| `05_sft_finetune.py` | Supervised Fine-Tuning (SFT) demonstrating prompt loss masking (`ignore_index=-1`) |
| `06_prepare_hf_dataset.py` | Downloads and formats `allenai/sciq` with `<|endoftext|>` stop tokens & held-out eval samples |
| `ui_server.py` | FastAPI backend for live GPU training, token confidence, and hallucination testing |
| `static/index.html` | Interactive single-page UI with high-school-level `ⓘ` info tooltips |
| `data/` | Ready-to-train `sciq` and `TinyShakespeare` tokenized datasets (~4.8 MB total) |

---

## VRAM & Scaling Intuition

For mixed-precision **AdamW training**, static memory is roughly **16 bytes per parameter** (`bf16` weights + `bf16` gradients + `fp32` master copy and Adam moments), plus activations:
- **~30M param SLM** (default config, `gpt2` 50k vocab): ~458 MB static + activations → **~3.2 GB peak VRAM** on an RTX 4060 (8 GB).
- **7B+ param LLM** (full pretraining): >110 GB static before activations → requires multi-GPU clusters or parameter-efficient fine-tuning (**LoRA / QLoRA**).

---

## License

MIT — see [LICENSE](LICENSE).
