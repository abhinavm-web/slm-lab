"""Stage 7: Abstention SFT + Direct Preference Optimization (DPO) from Scratch.

Teaches the pretrained Science SLM (checkpoints/sciq_ckpt.pt) WHEN to answer
and WHEN to say "I don't know based on the provided context."

Two-step alignment:
1. Abstention SFT: Fine-tunes on a balanced mix of grounded answerable Q&A (70%)
   and mismatched-context / unanswerable Q&A (30%) with prompt loss masking.
2. DPO (Direct Preference Optimization): Directly trains the policy model against
   a frozen reference model on (chosen, rejected) preference pairs so honest
   abstention beats plausible-sounding hallucinations.

Usage:
    python 07_abstention_sft_and_dpo.py
"""

import argparse
import copy
import importlib
import random
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F
from datasets import load_dataset

from config import PipelineConfig

model_mod = importlib.import_module("02_model")
GPT = model_mod.GPT
GPTConfig = model_mod.GPTConfig

ABSTAIN_TEXT = "I don't know based on the provided context."


def build_abstention_datasets(
    n_pairs: int = 600, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Build SFT examples and DPO (prompt, chosen, rejected) pairs from allenai/sciq."""
    rng = random.Random(seed)
    ds = load_dataset("allenai/sciq", split="train")
    valid_rows = [r for r in ds if (r.get("support") or "").strip()]
    rng.shuffle(valid_rows)

    sft_examples: list[dict] = []
    dpo_pairs: list[dict] = []

    for i in range(min(n_pairs, len(valid_rows) - 1)):
        row = valid_rows[i]
        other = valid_rows[(i + 1) % len(valid_rows)]

        q = row["question"].strip()
        ans = row["correct_answer"].strip()
        distractor = row["distractor1"].strip()
        good_ctx = row["support"].strip()
        wrong_ctx = other["support"].strip()

        if i % 3 == 0:
            # Unanswerable: question does not match the provided context
            prompt = f"### Science Context:\n{wrong_ctx}\n\n### Question:\n{q}\n\n### Answer:\n"
            sft_examples.append({"prompt": prompt, "response": ABSTAIN_TEXT})
            dpo_pairs.append({"prompt": prompt, "chosen": ABSTAIN_TEXT, "rejected": ans})
            dpo_pairs.append({"prompt": prompt, "chosen": ABSTAIN_TEXT, "rejected": distractor})
        else:
            # Answerable: grounded in the matching context
            prompt = f"### Science Context:\n{good_ctx}\n\n### Question:\n{q}\n\n### Answer:\n"
            sft_examples.append({"prompt": prompt, "response": ans})
            dpo_pairs.append({"prompt": prompt, "chosen": ans, "rejected": distractor})
            dpo_pairs.append({"prompt": prompt, "chosen": ans, "rejected": ABSTAIN_TEXT})

    # Add a few closed-book fictional / impossible questions
    for trick_q, fake_ans in [
        ("What is the melting point of vibranium in degrees Celsius?", "1538 degrees Celsius"),
        ("Which planet in the Andromeda galaxy has liquid nitrogen oceans?", "Kepler-442b"),
        ("What is the atomic number of unobtanium?", "119"),
    ]:
        p = f"### Question:\n{trick_q}\n\n### Answer:\n"
        sft_examples.append({"prompt": p, "response": ABSTAIN_TEXT})
        dpo_pairs.append({"prompt": p, "chosen": ABSTAIN_TEXT, "rejected": fake_ans})

    return sft_examples, dpo_pairs


def encode_pair(
    prompt: str, response: str, enc: tiktoken.Encoding, block_size: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    p_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    r_ids = enc.encode(response, allowed_special={"<|endoftext|>"}) + [enc.eot_token]
    full = p_ids + r_ids
    if len(full) > block_size + 1:
        # Keep the end of the prompt + full response if context is long
        overflow = len(full) - (block_size + 1)
        p_ids = p_ids[overflow:]
        full = p_ids + r_ids
    if len(p_ids) < 2 or len(full) > block_size + 1:
        return None

    x = full[:-1]
    y = full[1:]
    mask_len = len(p_ids) - 1
    y = [-1 if idx < mask_len else tok for idx, tok in enumerate(y)]

    pad_len = block_size - len(x)
    x = x + [0] * pad_len
    y = y + [-1] * pad_len
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def sequence_logprobs(
    model: GPT, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Compute sum of log-probabilities over non-masked (-1) response tokens."""
    logits, _ = model(x, y)
    log_probs = F.log_softmax(logits, dim=-1)
    mask = (y != -1)
    safe_y = y.masked_fill(~mask, 0)
    per_tok = torch.gather(log_probs, dim=-1, index=safe_y.unsqueeze(-1)).squeeze(-1)
    return (per_tok * mask).sum(dim=-1)


def main() -> None:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description="Abstention SFT + DPO training.")
    parser.add_argument("--base-ckpt", type=Path, default=cfg.checkpoint_dir / "sciq_ckpt.pt")
    parser.add_argument("--sft-steps", type=int, default=180)
    parser.add_argument("--dpo-steps", type=int, default=120)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL penalty strength.")
    args = parser.parse_args()

    if not args.base_ckpt.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found: {args.base_ckpt}. Train sciq_ckpt.pt first."
        )

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    gpt_cfg = GPTConfig(**ckpt["model_config"])
    meta = ckpt["meta"]
    enc = tiktoken.get_encoding("gpt2")

    sft_raw, dpo_raw = build_abstention_datasets(n_pairs=600)

    # --- Phase 1: Abstention SFT ---
    sft_tensors = [
        t for ex in sft_raw
        if (t := encode_pair(ex["prompt"], ex["response"], enc, gpt_cfg.block_size)) is not None
    ]
    sft_x = torch.stack([t[0] for t in sft_tensors]).to(device)
    sft_y = torch.stack([t[1] for t in sft_tensors]).to(device)

    policy = GPT(gpt_cfg).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.train()

    opt_sft = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=0.01)
    print(f"[phase 1: abstention SFT] {len(sft_x)} examples, {args.sft_steps} steps...")
    for step in range(1, args.sft_steps + 1):
        ix = torch.randint(0, len(sft_x), (16,), device=device)
        _, loss = policy(sft_x[ix], sft_y[ix])
        opt_sft.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt_sft.step()
        if step % 45 == 0 or step == 1 or step == args.sft_steps:
            print(f"  sft step {step:3d}/{args.sft_steps} | loss {loss.item():.4f}")

    # --- Phase 2: DPO (Direct Preference Optimization) ---
    ref_model = copy.deepcopy(policy).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    dpo_tensors = []
    for pair in dpo_raw:
        c = encode_pair(pair["prompt"], pair["chosen"], enc, gpt_cfg.block_size)
        r = encode_pair(pair["prompt"], pair["rejected"], enc, gpt_cfg.block_size)
        if c is not None and r is not None:
            dpo_tensors.append((c[0], c[1], r[0], r[1]))

    cx = torch.stack([t[0] for t in dpo_tensors]).to(device)
    cy = torch.stack([t[1] for t in dpo_tensors]).to(device)
    rx = torch.stack([t[2] for t in dpo_tensors]).to(device)
    ry = torch.stack([t[3] for t in dpo_tensors]).to(device)

    opt_dpo = torch.optim.AdamW(policy.parameters(), lr=3e-5, weight_decay=0.01)
    print(f"\n[phase 2: DPO] {len(cx)} preference pairs, {args.dpo_steps} steps (beta={args.beta})...")
    for step in range(1, args.dpo_steps + 1):
        ix = torch.randint(0, len(cx), (8,), device=device)
        pi_c = sequence_logprobs(policy, cx[ix], cy[ix])
        pi_r = sequence_logprobs(policy, rx[ix], ry[ix])
        with torch.no_grad():
            ref_c = sequence_logprobs(ref_model, cx[ix], cy[ix])
            ref_r = sequence_logprobs(ref_model, rx[ix], ry[ix])

        chosen_rewards = args.beta * (pi_c - ref_c)
        rejected_rewards = args.beta * (pi_r - ref_r)
        logits_diff = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(logits_diff).mean()

        opt_dpo.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt_dpo.step()

        if step % 30 == 0 or step == 1 or step == args.dpo_steps:
            margin = logits_diff.mean().item()
            acc = (logits_diff > 0).float().mean().item() * 100.0
            print(
                f"  dpo step {step:3d}/{args.dpo_steps} | loss {loss.item():.4f} | "
                f"reward margin {margin:+.3f} | pref acc {acc:.1f}%"
            )

    out_path = cfg.checkpoint_dir / "sciq_abstain_dpo.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "model_config": gpt_cfg.__dict__,
            "meta": meta,
            "step": args.sft_steps + args.dpo_steps,
            "best_val_loss": float(ckpt.get("best_val_loss", 0.0)),
        },
        out_path,
    )
    print(f"\n[done] Saved abstention + DPO checkpoint to {out_path}")


if __name__ == "__main__":
    main()
