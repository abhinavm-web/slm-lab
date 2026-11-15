"""Stage 3: Pretraining Loop from Scratch (Next-Token Prediction).

Trains the GPT model on train.bin and monitors generalization on val.bin.
Logs loss, perplexity, throughput (tokens/sec), and peak GPU VRAM.

Usage:
    python 03_train.py
    python 03_train.py --max-iters 1000
"""

import argparse
import importlib
import json
import math
import time

import numpy as np
import torch

from config import PipelineConfig

model_mod = importlib.import_module("02_model")
GPT = model_mod.GPT
GPTConfig = model_mod.GPTConfig


def get_batch(data: np.memmap, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(it: int, cfg: PipelineConfig) -> float:
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it >= cfg.max_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: np.memmap,
    val_data: np.memmap,
    cfg: PipelineConfig,
    device: str,
    ctx: torch.amp.autocast,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.batch_size, cfg.block_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def configure_optimizer(model: GPT, cfg: PipelineConfig) -> torch.optim.AdamW:
    # Apply weight decay only to 2D+ weight matrices (not LayerNorm scales or 1D biases)
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=cfg.learning_rate, betas=(0.9, 0.95))


def main() -> None:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description="Pretrain Small Language Model from scratch.")
    parser.add_argument("--max-iters", type=int, default=cfg.max_iters)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    args = parser.parse_args()
    cfg.max_iters = args.max_iters
    cfg.batch_size = args.batch_size

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[hw] Using GPU: {gpu_name}")
    else:
        print("[hw] WARNING: CUDA not available, training on CPU (will be slow).")

    meta = json.loads((cfg.data_dir / "meta.json").read_text(encoding="utf-8"))
    train_data = np.memmap(cfg.data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(cfg.data_dir / "val.bin", dtype=np.uint16, mode="r")

    gpt_cfg = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    model = GPT(gpt_cfg).to(device)
    optimizer = configure_optimizer(model, cfg)

    pt_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.dtype]
    if device == "cpu":
        pt_dtype = torch.float32
    ctx = torch.amp.autocast(device_type=device, dtype=pt_dtype, enabled=(device == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and pt_dtype == torch.float16))

    tokens_per_step = cfg.batch_size * cfg.grad_accum_steps * cfg.block_size
    expected_init_loss = math.log(gpt_cfg.vocab_size)
    print(
        f"[model] Params: {model.count_parameters():,} ({model.count_parameters()/1e6:.2f}M) | "
        f"Vocab: {gpt_cfg.vocab_size:,} | Context: {cfg.block_size}"
    )
    print(
        f"[train] Steps: {cfg.max_iters} | Tokens/step: {tokens_per_step:,} | "
        f"Random baseline loss ln(V) = {expected_init_loss:.2f}"
    )

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []
    t_start = time.perf_counter()

    for it in range(cfg.max_iters + 1):
        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            losses = estimate_loss(model, train_data, val_data, cfg, device, ctx)
            elapsed = time.perf_counter() - t_start
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
            ppl = math.exp(min(losses["val"], 20.0))
            print(
                f"step {it:4d}/{cfg.max_iters} | "
                f"train loss {losses['train']:.3f} | val loss {losses['val']:.3f} (ppl {ppl:.1f}) | "
                f"lr {get_lr(it, cfg):.2e} | vram {vram_mb:.0f} MB | {elapsed:.1f}s"
            )
            history.append({"step": it, **losses, "val_ppl": ppl, "elapsed_s": elapsed, "vram_mb": vram_mb})

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_config": gpt_cfg.__dict__,
                        "meta": meta,
                        "step": it,
                        "best_val_loss": best_val_loss,
                    },
                    cfg.checkpoint_dir / "ckpt.pt",
                )

        if it == cfg.max_iters:
            break

        lr = get_lr(it, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum_steps):
            x, y = get_batch(train_data, cfg.batch_size, cfg.block_size, device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    total_time = time.perf_counter() - t_start
    total_tokens = cfg.max_iters * tokens_per_step
    (cfg.checkpoint_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(
        f"\n[done] Finished {cfg.max_iters} steps ({total_tokens:,} tokens) in {total_time:.1f}s "
        f"({total_tokens / total_time:,.0f} tok/s). Best val loss: {best_val_loss:.3f}"
    )


if __name__ == "__main__":
    main()
