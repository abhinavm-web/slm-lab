"""Stage 8: Automated Evaluation & Calibration Suite ("Evals").

Benchmarks any trained checkpoint on held-out Science Q&A (sciq) examples
across three modes:
1. Grounded (Answerable): Correct context provided -> measures Grounded Accuracy.
2. Closed-Book: No context provided -> measures Closed-Book Accuracy & Hallucination Rate.
3. Mismatched Context (Unanswerable): Wrong context provided -> measures Abstention Accuracy.

Also measures Confidence Calibration (avg token confidence when right vs. when wrong).

Usage:
    python 08_eval_suite.py --ckpt checkpoints/sciq_abstain_dpo.pt --n 25
"""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from config import PipelineConfig

model_mod = importlib.import_module("02_model")
GPT = model_mod.GPT
GPTConfig = model_mod.GPTConfig

BASE_DIR = Path(__file__).resolve().parent
EVAL_PATH = BASE_DIR / "data" / "sciq" / "eval_samples.json"


def _is_abstain(text: str) -> bool:
    t = text.strip().lower()
    return "don't know" in t or "do not know" in t or "not enough information" in t


def _greedy_generate_with_conf(
    model: GPT,
    prompt: str,
    enc: tiktoken.Encoding,
    block_size: int,
    device: str,
    max_tokens: int = 32,
) -> tuple[str, float]:
    ids = enc.encode(prompt, allowed_special={"<|endoftext|>"}) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    probs_list: list[float] = []
    out_ids: list[int] = []

    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.argmax(logits, dim=-1).item())
            if next_id == enc.eot_token:
                break
            probs_list.append(float(probs[0, next_id].item()) * 100.0)
            out_ids.append(next_id)
            idx = torch.cat((idx, torch.tensor([[next_id]], dtype=torch.long, device=device)), dim=1)

    text = enc.decode(out_ids).strip()
    avg_conf = round(float(np.mean(probs_list)) if probs_list else 0.0, 1)
    return text, avg_conf


def evaluate_checkpoint(ckpt_path: Path, n_samples: int = 25, dataset: str = "sciq") -> dict:
    eval_path = BASE_DIR / "data" / dataset / "eval_samples.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {eval_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gpt_cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(gpt_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    samples = json.loads(eval_path.read_text(encoding="utf-8"))[:n_samples]

    grounded_ok = 0
    closed_ok = 0
    abstain_ok = 0
    hallucinations = 0
    right_confs: list[float] = []
    wrong_confs: list[float] = []
    rows: list[dict] = []

    for i, s in enumerate(samples):
        q = s["question"]
        truth = s["correct_answer"]
        truth_l = truth.lower()
        good_ctx = s["support"]
        wrong_ctx = samples[(i + 1) % len(samples)]["support"]

        # 1. Grounded
        g_prompt = f"### Science Context:\n{good_ctx}\n\n### Question:\n{q}\n\n### Answer:\n"
        g_ans, g_conf = _greedy_generate_with_conf(model, g_prompt, enc, gpt_cfg.block_size, device)
        g_match = truth_l in g_ans.lower()
        if g_match:
            grounded_ok += 1
            right_confs.append(g_conf)
        else:
            wrong_confs.append(g_conf)

        # 2. Closed-Book
        c_prompt = f"### Question:\n{q}\n\n### Answer:\n"
        c_ans, c_conf = _greedy_generate_with_conf(model, c_prompt, enc, gpt_cfg.block_size, device)
        c_match = truth_l in c_ans.lower()
        c_abs = _is_abstain(c_ans)
        if c_match:
            closed_ok += 1
        elif not c_abs:
            hallucinations += 1

        # 3. Mismatched Context (Unanswerable -> should abstain)
        m_prompt = f"### Science Context:\n{wrong_ctx}\n\n### Question:\n{q}\n\n### Answer:\n"
        m_ans, m_conf = _greedy_generate_with_conf(model, m_prompt, enc, gpt_cfg.block_size, device)
        m_abs = _is_abstain(m_ans)
        if m_abs:
            abstain_ok += 1
            right_confs.append(m_conf)
        else:
            hallucinations += 1
            wrong_confs.append(m_conf)

        if i < 10:
            rows.append(
                {
                    "question": q,
                    "truth": truth,
                    "grounded_ans": g_ans,
                    "grounded_ok": g_match,
                    "mismatched_ans": m_ans,
                    "abstained_ok": m_abs,
                }
            )

    total = max(1, len(samples))
    return {
        "checkpoint": ckpt_path.name,
        "n_samples": len(samples),
        "grounded_accuracy_pct": round(grounded_ok / total * 100.0, 1),
        "closed_book_accuracy_pct": round(closed_ok / total * 100.0, 1),
        "abstention_accuracy_pct": round(abstain_ok / total * 100.0, 1),
        "hallucination_rate_pct": round(hallucinations / (total * 2) * 100.0, 1),
        "avg_conf_when_right_pct": round(float(np.mean(right_confs)) if right_confs else 0.0, 1),
        "avg_conf_when_wrong_pct": round(float(np.mean(wrong_confs)) if wrong_confs else 0.0, 1),
        "sample_rows": rows,
    }


if __name__ == "__main__":
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description="Run automated evals on a checkpoint.")
    parser.add_argument("--ckpt", type=Path, default=cfg.checkpoint_dir / "sciq_abstain_dpo.pt")
    parser.add_argument("--n", type=int, default=25)
    args = parser.parse_args()

    report = evaluate_checkpoint(args.ckpt, n_samples=args.n)
    print(json.dumps({k: v for k, v in report.items() if k != "sample_rows"}, indent=2))
