"""Stage 4: Inference & Sampling (Autoregressive Generation).

Loads a trained checkpoint and generates text token-by-token.
Demonstrates how temperature, top-k, and top-p (nucleus) sampling
control determinism vs. creativity.

Usage:
    python 04_generate.py --prompt "ROMEO:"
    python 04_generate.py --prompt "ROMEO:" --temperature 0.2 --top-k 10
    python 04_generate.py --ckpt checkpoints/sft_ckpt.pt --prompt "### Instruction:\nWhat is overfitting?\n\n### Response:\n"
"""

import argparse
import importlib
from pathlib import Path

import torch

from config import PipelineConfig

model_mod = importlib.import_module("02_model")
GPT = model_mod.GPT
GPTConfig = model_mod.GPTConfig


def encode_prompt(prompt: str, meta: dict) -> list[int]:
    if meta["tokenizer_type"] == "gpt2":
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return enc.encode_ordinary(prompt)
    stoi = meta["stoi"]
    return [stoi[c] for c in prompt if c in stoi]


def decode_tokens(ids: list[int], meta: dict) -> str:
    if meta["tokenizer_type"] == "gpt2":
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return enc.decode(ids)
    itos = {int(k): v for k, v in meta["itos"].items()}
    return "".join(itos.get(i, "") for i in ids)


def main() -> None:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description="Generate text from a trained SLM checkpoint.")
    parser.add_argument("--ckpt", type=Path, default=cfg.checkpoint_dir / "ckpt.pt")
    parser.add_argument("--prompt", type=str, default="FIRST CITIZEN:\n")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    gpt_cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(gpt_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    meta = ckpt["meta"]
    prompt_ids = encode_prompt(args.prompt, meta)
    if not prompt_ids:
        prompt_ids = [0]
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    print("=" * 60)
    print(f"Checkpoint: {args.ckpt.name} (step {ckpt.get('step', '?')}, val_loss {ckpt.get('best_val_loss', 0):.3f})")
    print(f"Sampling:   temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("=" * 60)
    print(decode_tokens(out[0].tolist(), meta))
    print("=" * 60)


if __name__ == "__main__":
    main()
