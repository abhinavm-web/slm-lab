"""Stage 1: Data Preparation & Tokenization.

Turns raw text into flat binary arrays of integer token IDs (train.bin / val.bin)
plus a meta.json describing the vocabulary.

Usage:
    python 01_prepare_data.py
    python 01_prepare_data.py --tokenizer char
    python 01_prepare_data.py --input path/to/my_corpus.txt
"""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

from config import PipelineConfig

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_default_corpus(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[data] Using cached corpus: {dest} ({dest.stat().st_size:,} bytes)")
        return
    print(f"[data] Downloading TinyShakespeare to {dest}...")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, dest)
    print(f"[data] Downloaded {dest.stat().st_size:,} bytes.")


def encode_with_gpt2(text: str) -> tuple[list[int], dict]:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(text)
    sample = "Applied AI requires understanding compute, data, and evaluation."
    sample_tokens = [enc.decode([t]) for t in enc.encode_ordinary(sample)]
    print(f"[tokenizer] Sample split ({len(sample_tokens)} tokens): {sample_tokens}")
    meta = {
        "tokenizer_type": "gpt2",
        "vocab_size": enc.n_vocab,
    }
    return ids, meta


def encode_with_char(text: str) -> tuple[list[int], dict]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    ids = [stoi[c] for c in text]
    meta = {
        "tokenizer_type": "char",
        "vocab_size": len(chars),
        "stoi": stoi,
        "itos": itos,
    }
    return ids, meta


def main() -> None:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description="Prepare and tokenize training data.")
    parser.add_argument("--input", type=Path, default=cfg.data_dir / "input.txt")
    parser.add_argument(
        "--tokenizer",
        choices=["gpt2", "char"],
        default=cfg.tokenizer_type,
        help="Subword BPE (gpt2) or character-level (char).",
    )
    args = parser.parse_args()

    if args.input == cfg.data_dir / "input.txt":
        download_default_corpus(args.input)

    text = args.input.read_text(encoding="utf-8")
    print(f"[data] Corpus length: {len(text):,} characters")

    if args.tokenizer == "gpt2":
        ids, meta = encode_with_gpt2(text)
    else:
        ids, meta = encode_with_char(text)

    n = len(ids)
    split_idx = int(n * 0.9)
    train_ids = np.array(ids[:split_idx], dtype=np.uint16)
    val_ids = np.array(ids[split_idx:], dtype=np.uint16)

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    train_ids.tofile(cfg.data_dir / "train.bin")
    val_ids.tofile(cfg.data_dir / "val.bin")

    meta["train_tokens"] = len(train_ids)
    meta["val_tokens"] = len(val_ids)
    (cfg.data_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    chars_per_token = len(text) / n
    print(f"[tokenizer] Type: {meta['tokenizer_type']} | Vocab size: {meta['vocab_size']:,}")
    print(f"[tokenizer] Total tokens: {n:,} ({chars_per_token:.2f} chars/token)")
    print(f"[data] Saved train.bin ({len(train_ids):,} tokens) & val.bin ({len(val_ids):,} tokens)")


if __name__ == "__main__":
    main()
