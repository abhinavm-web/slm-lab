"""Stage 10: LoRA (Low-Rank Adaptation) Fine-Tuning from Scratch.

Demonstrates how enterprises fine-tune large models on a single GPU:
1. Freeze 99%+ of the pretrained base model weights.
2. Inject tiny trainable low-rank matrices A (in -> r) and B (r -> out) into
   attention projections: output = W*x + (alpha/r) * B*(A*x).
3. Train ONLY A and B, then merge them back into W for zero-overhead inference.

Usage:
    python 10_lora_finetune.py
"""

import argparse
import importlib
import math
from pathlib import Path

import tiktoken
import torch
import torch.nn as nn
from datasets import load_dataset

from config import PipelineConfig

model_mod = importlib.import_module("02_model")
GPT = model_mod.GPT
GPTConfig = model_mod.GPTConfig


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with trainable low-rank matrices A and B."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(base.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, base.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scale
        return base_out + lora_out

    def merge_into_base(self) -> nn.Linear:
        with torch.no_grad():
            delta = (self.lora_A @ self.lora_B).T * self.scale
            self.base.weight.add_(delta)
        return self.base


def apply_lora(model: GPT, r: int = 8, alpha: int = 16) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for block in model.transformer.h:
        block.attn.c_attn = LoRALinear(block.attn.c_attn, r=r, alpha=alpha)
        block.attn.c_proj = LoRALinear(block.attn.c_proj, r=r, alpha=alpha)


def merge_lora(model: GPT) -> None:
    for block in model.transformer.h:
        if isinstance(block.attn.c_attn, LoRALinear):
            block.attn.c_attn = block.attn.c_attn.merge_into_base()
        if isinstance(block.attn.c_proj, LoRALinear):
            block.attn.c_proj = block.attn.c_proj.merge_into_base()


def main() -> None:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description="LoRA fine-tuning from scratch.")
    parser.add_argument("--base-ckpt", type=Path, default=cfg.checkpoint_dir / "sciq_ckpt.pt")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    gpt_cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(gpt_cfg)
    model.load_state_dict(ckpt["model"])

    apply_lora(model, r=args.rank, alpha=args.alpha)
    model = model.to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[lora] rank={args.rank}, alpha={args.alpha} | "
        f"Total params: {total_p:,} | Trainable LoRA params: {train_p:,} ({train_p / total_p * 100:.2f}%)"
    )

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("allenai/sciq", split="train[:400]")
    xs, ys = [], []
    for row in ds:
        sup = (row.get("support") or "").strip()
        if not sup:
            continue
        p_ids = enc.encode(f"### Science Context:\n{sup}\n\n### Question:\n{row['question'].strip()}\n\n### Answer:\n")
        r_ids = enc.encode(row["correct_answer"].strip()) + [enc.eot_token]
        full = (p_ids + r_ids)[: gpt_cfg.block_size + 1]
        if len(full) < 4 or len(p_ids) >= len(full) - 1:
            continue
        x = full[:-1]
        y = [-1 if i < len(p_ids) - 1 else tok for i, tok in enumerate(full[1:])]
        pad = gpt_cfg.block_size - len(x)
        xs.append(torch.tensor(x + [0] * pad, dtype=torch.long))
        ys.append(torch.tensor(y + [-1] * pad, dtype=torch.long))

    X = torch.stack(xs).to(device)
    Y = torch.stack(ys).to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    model.train()
    for step in range(1, args.steps + 1):
        ix = torch.randint(0, len(X), (16,), device=device)
        _, loss = model(X[ix], Y[ix])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 30 == 0 or step == 1 or step == args.steps:
            print(f"  lora step {step:3d}/{args.steps} | loss {loss.item():.4f}")

    merge_lora(model)
    out_path = cfg.checkpoint_dir / "sciq_lora_merged.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": gpt_cfg.__dict__,
            "meta": ckpt["meta"],
            "step": args.steps,
            "best_val_loss": float(ckpt.get("best_val_loss", 0.0)),
        },
        out_path,
    )
    print(f"[lora] Merged adapters into base weights and saved {out_path}")


if __name__ == "__main__":
    main()
