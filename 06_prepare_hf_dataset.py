"""Stage 6: Prepare a Domain Dataset from Hugging Face (allenai/sciq).

Downloads the Science Q&A dataset (Physics, Chemistry, Biology, Earth Science),
formats each example with its textbook support passage + question + answer + EOS token,
and writes train.bin, val.bin, meta.json, and eval_samples.json.

Usage:
    python 06_prepare_hf_dataset.py
"""

import json
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset


def format_sciq_example(row: dict) -> str:
    support = (row.get("support") or "").strip()
    question = row["question"].strip()
    answer = row["correct_answer"].strip()
    if support:
        return (
            f"### Science Context:\n{support}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Answer:\n{answer}<|endoftext|>\n\n"
        )
    return f"### Question:\n{question}\n\n### Answer:\n{answer}<|endoftext|>\n\n"


def prepare_sciq(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("allenai/sciq")
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256

    train_ids: list[int] = []
    for row in ds["train"]:
        text = format_sciq_example(row)
        train_ids.extend(enc.encode(text, allowed_special={"<|endoftext|>"}))

    val_ids: list[int] = []
    eval_samples: list[dict] = []
    for i, row in enumerate(ds["validation"]):
        text = format_sciq_example(row)
        val_ids.extend(enc.encode(text, allowed_special={"<|endoftext|>"}))
        if i < 100:
            eval_samples.append(
                {
                    "id": i,
                    "question": row["question"].strip(),
                    "correct_answer": row["correct_answer"].strip(),
                    "distractors": [
                        row["distractor1"].strip(),
                        row["distractor2"].strip(),
                        row["distractor3"].strip(),
                    ],
                    "support": (row.get("support") or "").strip(),
                }
            )

    np.array(train_ids, dtype=np.uint16).tofile(out_dir / "train.bin")
    np.array(val_ids, dtype=np.uint16).tofile(out_dir / "val.bin")
    (out_dir / "eval_samples.json").write_text(json.dumps(eval_samples, indent=2), encoding="utf-8")

    meta = {
        "dataset_name": "allenai/sciq",
        "domain": "Science (Physics, Chemistry, Biology, Earth Science)",
        "tokenizer_type": "gpt2",
        "vocab_size": enc.n_vocab,
        "eot_token": eot,
        "train_examples": len(ds["train"]),
        "val_examples": len(ds["validation"]),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    base = Path(__file__).resolve().parent / "data" / "sciq"
    info = prepare_sciq(base)
    print("[sciq] Prepared dataset:")
    print(json.dumps(info, indent=2))
