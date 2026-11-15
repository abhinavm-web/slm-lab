"""Central configuration for the from-scratch Small Language Model (SLM) pipeline.

Designed for an 8 GB VRAM GPU (e.g., NVIDIA RTX 4060).
Every hyperparameter is documented with both its technical role and its
practical trade-off (memory, speed, quality).
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    # --- Paths ---
    project_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = project_dir / "data"
    checkpoint_dir: Path = project_dir / "checkpoints"

    # --- Tokenization ---
    # "gpt2" uses tiktoken's 50,257-token BPE vocabulary (industry-standard subword BPE).
    # "char" uses a character-level vocabulary (~65 tokens) for fast, zero-dependency intuition.
    tokenizer_type: str = "gpt2"

    # --- Model Architecture (~10M - 30M parameters) ---
    # Context window: max tokens the model can attend to at once.
    # Attention memory scales O(block_size^2), so doubling this 4x's attention memory.
    block_size: int = 256
    n_layer: int = 6       # Number of Transformer blocks stacked vertically
    n_head: int = 6        # Attention heads per block (n_embd must be divisible by n_head)
    n_embd: int = 384      # Width of the residual stream / hidden dimension
    dropout: float = 0.1   # Regularization to prevent memorizing small datasets
    bias: bool = False     # Modern LLMs (LLaMA, etc.) omit bias in Linear/LayerNorm layers

    # --- Training Loop ---
    batch_size: int = 16           # Sequences per micro-step
    grad_accum_steps: int = 4      # Effective batch size = 16 * 4 = 64 sequences (~16,384 tokens/step)
    max_iters: int = 600           # Total optimizer steps (fast ~1-2 min run on RTX 4060)
    eval_interval: int = 100       # Evaluate train/val loss every N steps
    eval_iters: int = 25           # Number of batches averaged per evaluation

    # --- Optimizer & Learning Rate Schedule ---
    learning_rate: float = 6e-4    # Peak learning rate
    min_lr: float = 6e-5           # Cosine decay floor (typically 10% of peak LR)
    warmup_iters: int = 50         # Linear warmup steps to stabilize early gradients
    weight_decay: float = 0.1      # L2-like regularization on 2D weights
    grad_clip: float = 1.0         # Prevents exploding gradients

    # --- Hardware / Precision ---
    # RTX 4060 (Ada Lovelace) natively supports bfloat16, which has the dynamic range
    # of fp32 with half the memory and ~2x Tensor Core throughput.
    dtype: str = "bfloat16"

    # --- SFT (Post-Training) ---
    sft_max_iters: int = 150
    sft_learning_rate: float = 1e-4
    sft_batch_size: int = 8
