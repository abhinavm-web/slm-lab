# Small Language Model (SLM) Training Pipeline — From Scratch

An educational, zero-magic PyTorch pipeline built to run on a local **NVIDIA RTX 4060 (8 GB VRAM)** and teach every fundamental stage of training a modern Large/Small Language Model.

## Pipeline Stages & Executive Takeaways

| Stage | Script | Engineering Concept | Executive / Applied AI Takeaway |
|---|---|---|---|
| **1. Data & Tokenization** | `01_prepare_data.py` | Raw text → BPE subword IDs (`tiktoken`), train/val split | Tokenization efficiency (chars/token) directly drives API cost, latency, and effective context window. |
| **2. Architecture** | `02_model.py` | Decoder-only Transformer (Embeddings, Causal Self-Attention, MLP, Weight Tying) | Parameter count + context length determine VRAM; attention memory scales quadratically $O(T^2)$ with context length. |
| **3. Pretraining** | `03_train.py` | Next-token prediction, Cross-Entropy loss, AdamW, Cosine LR, `bfloat16` AMP, Grad Accumulation | Pretraining teaches language & patterns from raw text; watch `val_loss` vs `train_loss` to catch memorization/overfitting. |
| **4. Inference** | `04_generate.py` | Autoregressive decoding, Temperature, Top-k, Top-p (Nucleus) sampling | Sampling settings trade determinism (factual/code) for diversity (brainstorming) without retraining. |
| **5. Post-Training (SFT)** | `05_sft_finetune.py` | Instruction/Response formatting + **Prompt Loss Masking** (`ignore_index=-1`) | Pretraining creates a "text completer"; SFT turns it into an instruction-following assistant. |

## Quickstart

```powershell
# Activate the CUDA virtual environment
.\.venv\Scripts\Activate.ps1

# 1. Prepare & tokenize data (downloads TinyShakespeare ~1.1 MB)
python 01_prepare_data.py

# 2. Inspect model parameter count & VRAM estimate
python 02_model.py

# 3. Pretrain on GPU (~1-2 minutes on RTX 4060)
python 03_train.py

# 4. Generate text from the pretrained base model
python 04_generate.py --prompt "FIRST CITIZEN:\n"

# 5. Run Supervised Fine-Tuning (SFT) & test instruction following
python 05_sft_finetune.py
python 04_generate.py --ckpt checkpoints/sft_ckpt.pt --prompt "### Instruction:\nSummarize the duty of a wise ruler.\n\n### Response:\n"
```

## VRAM & Scaling Intuition (Why 8 GB Trains ~30M Params, Not 7B)

For full-precision / mixed-precision **AdamW training**, static memory is roughly **16 bytes per parameter**:
- Weights (`bf16`): 2 bytes
- Gradients (`bf16`): 2 bytes
- Optimizer states (`fp32` master copy + Adam `m` + `v`): 12 bytes

So:
- **~30M param SLM** (this repo with `gpt2` vocab): ~480 MB static + activations → fits easily in **8 GB VRAM**.
- **7B param LLM** (full pretraining): ~112 GB static *before* activations → requires multi-GPU clusters (H100s) or parameter-efficient fine-tuning (**QLoRA**).
