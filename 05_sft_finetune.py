"""Stage 5: Post-Training — Supervised Fine-Tuning (SFT).

Takes the pretrained base model (checkpoints/ckpt.pt) and fine-tunes it on
(Instruction, Response) pairs.

Key concept: LOSS MASKING.
We set target token IDs to -1 for the prompt/instruction tokens so cross-entropy
loss is computed ONLY on the assistant's response tokens.

Usage:
    python 05_sft_finetune.py
"""

import argparse
import importlib
import json
from pathlib import Path

import torch

from config import PipelineConfig

model_mod = importlib.import_module("02_model")
GPT = model_mod.GPT
GPTConfig = model_mod.GPTConfig

gen_mod = importlib.import_module("04_generate")
encode_prompt = gen_mod.encode_prompt

DEFAULT_SFT_EXAMPLES = [
    {
        "instruction": "Speak like a royal advisor on strategy.",
        "response": "My lord, weigh the cost of every march before the banners rise; swift counsel saves both crown and treasury.",
    },
    {
        "instruction": "Give a brief warning about haste in leadership.",
        "response": "Haste is a false friend to judgment; measure twice in silence ere you command in public.",
    },
    {
        "instruction": "Summarize the duty of a wise ruler.",
        "response": "A wise ruler listens to the quietest truth, rewards merit above flattery, and guards the realm against pride.",
    },
    {
        "instruction": "Compose two lines on learning and mastery.",
        "response": "First learn the craft in small and humble parts, / Then command the whole with steady, practiced arts.",
    },
    {
        "instruction": "What separates knowledge from wisdom?",
        "response": "Knowledge counts the stones upon the road; wisdom knows which path leads safely home.",
    },
    {
        "instruction": "Advise a captain before a difficult campaign.",
        "response": "Keep thy supply lines closer than thy sword, and know the ground before the dawn breaks.",
    },
]


def build_sft_tensors(
    examples: list[dict[str, str]], meta: dict, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = 0
    xs, ys = [], []
    for ex in examples:
        prompt_text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n"
        resp_text = f"{ex['response']}\n"

        prompt_ids = encode_prompt(prompt_text, meta)
        resp_ids = encode_prompt(resp_text, meta)

        full_ids = (prompt_ids + resp_ids)[: block_size + 1]
        # Next-token prediction: input is [:-1], target is [1:]
        x = full_ids[:-1]
        y = full_ids[1:]

        # Mask out prompt tokens in target (-1 is ignored by F.cross_entropy)
        prompt_mask_len = max(0, len(prompt_ids) - 1)
        y = [-1 if i < prompt_mask_len else tok for i, tok in enumerate(y)]

        # Pad to block_size
        pad_len = block_size - len(x)
        x = x + [pad_id] * pad_len
        y = y + [-1] * pad_len

        xs.append(torch.tensor(x, dtype=torch.long))
        ys.append(torch.tensor(y, dtype=torch.long))

    return torch.stack(xs), torch.stack(ys)


def main() -> None:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) on top of base checkpoint.")
    parser.add_argument("--base-ckpt", type=Path, default=cfg.checkpoint_dir / "ckpt.pt")
    parser.add_argument("--sft-data", type=Path, default=None, help="Optional JSONL with instruction/response.")
    parser.add_argument("--iters", type=int, default=cfg.sft_max_iters)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    gpt_cfg = GPTConfig(**ckpt["model_config"])
    meta = ckpt["meta"]

    model = GPT(gpt_cfg).to(device)
    model.load_state_dict(ckpt["model"])

    if args.sft_data and args.sft_data.exists():
        examples = [json.loads(line) for line in args.sft_data.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        examples = DEFAULT_SFT_EXAMPLES

    X, Y = build_sft_tensors(examples, meta, gpt_cfg.block_size)
    X, Y = X.to(device), Y.to(device)
    supervised_tokens = int((Y != -1).sum().item())
    print(f"[sft] Loaded {len(examples)} examples ({supervised_tokens} supervised response tokens; prompt tokens masked)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.sft_learning_rate, weight_decay=0.01)
    model.train()

    for it in range(1, args.iters + 1):
        ix = torch.randint(0, len(X), (min(cfg.sft_batch_size, len(X)),), device=device)
        _, loss = model(X[ix], Y[ix])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % 30 == 0 or it == 1 or it == args.iters:
            print(f"[sft] step {it:3d}/{args.iters} | response loss {loss.item():.4f}")

    out_path = cfg.checkpoint_dir / "sft_ckpt.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": gpt_cfg.__dict__,
            "meta": meta,
            "step": args.iters,
            "best_val_loss": loss.item(),
        },
        out_path,
    )
    print(f"[sft] Saved fine-tuned checkpoint to {out_path}")


if __name__ == "__main__":
    main()
