"""Stage 6: Prepare Real-Industry Domain Datasets from Hugging Face.

Supports four domain datasets, each formatted into:
- train.bin / val.bin (tokenized with gpt2 BPE + <|endoftext|> stop token)
- meta.json (vocabulary & token counts)
- eval_samples.json (held-out Q&A + context for Hallucination Lab & Automated Evals)
- rag_docs.json (unique domain passages for RAG retrieval)

Domains:
- sciq:       Science Q&A (allenai/sciq)
- finance:    SEC 10-K Filings Q&A (virattt/financial-qa-10K)
- healthcare: Biomedical Research Q&A (qiaojin/PubMedQA, pqa_labeled)
- legal:      SEC Material Contract Clauses (coastalcph/lex_glue, ledgar)

Usage:
    python 06_prepare_hf_dataset.py --domain all
    python 06_prepare_hf_dataset.py --domain finance
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

from domain_catalog import DOMAIN_CATALOG

BASE_DATA_DIR = Path(__file__).resolve().parent / "data"


def _save_domain(
    out_dir: Path,
    dataset_name: str,
    domain_label: str,
    train_texts: list[str],
    val_texts: list[str],
    eval_samples: list[dict],
    rag_docs: list[str],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    train_ids: list[int] = []
    for t in train_texts:
        train_ids.extend(enc.encode(t, allowed_special={"<|endoftext|>"}))

    val_ids: list[int] = []
    for t in val_texts:
        val_ids.extend(enc.encode(t, allowed_special={"<|endoftext|>"}))

    np.array(train_ids, dtype=np.uint16).tofile(out_dir / "train.bin")
    np.array(val_ids, dtype=np.uint16).tofile(out_dir / "val.bin")
    (out_dir / "eval_samples.json").write_text(json.dumps(eval_samples[:100], indent=2), encoding="utf-8")
    (out_dir / "rag_docs.json").write_text(json.dumps(rag_docs), encoding="utf-8")

    meta = {
        "dataset_name": dataset_name,
        "domain": domain_label,
        "tokenizer_type": "gpt2",
        "vocab_size": enc.n_vocab,
        "eot_token": enc.eot_token,
        "train_examples": len(train_texts),
        "val_examples": len(val_texts),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def prepare_sciq() -> dict:
    ds = load_dataset("allenai/sciq")
    fmt = lambda r: (
        (f"### Science Context:\n{r['support'].strip()}\n\n" if (r.get("support") or "").strip() else "")
        + f"### Question:\n{r['question'].strip()}\n\n### Answer:\n{r['correct_answer'].strip()}<|endoftext|>\n\n"
    )
    train_texts = [fmt(r) for r in ds["train"]]
    val_texts = [fmt(r) for r in ds["validation"]]
    eval_samples = [
        {
            "id": i,
            "question": r["question"].strip(),
            "correct_answer": r["correct_answer"].strip(),
            "distractors": [r["distractor1"].strip(), r["distractor2"].strip(), r["distractor3"].strip()],
            "support": (r.get("support") or "").strip(),
        }
        for i, r in enumerate(ds["validation"].select(range(100)))
    ]
    rag_docs = list({(r.get("support") or "").strip() for r in ds["train"] if (r.get("support") or "").strip()})
    return _save_domain(
        BASE_DATA_DIR / "sciq",
        "allenai/sciq",
        "Science (Physics, Chemistry, Biology, Earth Science)",
        train_texts,
        val_texts,
        eval_samples,
        rag_docs,
    )


def prepare_finance() -> dict:
    ds = load_dataset("virattt/financial-qa-10K", split="train")
    split_idx = int(len(ds) * 0.9)
    train_rows = ds.select(range(split_idx))
    val_rows = ds.select(range(split_idx, len(ds)))

    def fmt(r: dict) -> str:
        ctx = (r.get("context") or "").strip()
        ticker = (r.get("ticker") or "").strip()
        prefix = f"[{ticker} 10-K] " if ticker else ""
        return (
            f"### Context:\n{prefix}{ctx}\n\n"
            f"### Question:\n{r['question'].strip()}\n\n"
            f"### Answer:\n{r['answer'].strip()}<|endoftext|>\n\n"
        )

    train_texts = [fmt(r) for r in train_rows]
    val_texts = [fmt(r) for r in val_rows]
    eval_samples = [
        {
            "id": i,
            "question": r["question"].strip(),
            "correct_answer": r["answer"].strip(),
            "distractors": [],
            "support": f"[{r.get('ticker', '')} 10-K] {(r.get('context') or '').strip()}".strip(),
        }
        for i, r in enumerate(val_rows)
        if i < 100
    ]
    rag_docs = list({(r.get("context") or "").strip() for r in train_rows if (r.get("context") or "").strip()})
    return _save_domain(
        BASE_DATA_DIR / "finance",
        "virattt/financial-qa-10K",
        "Finance (SEC 10-K Filings Q&A)",
        train_texts,
        val_texts,
        eval_samples,
        rag_docs,
    )


def prepare_healthcare() -> dict:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    split_idx = int(len(ds) * 0.85)
    train_rows = ds.select(range(split_idx))
    val_rows = ds.select(range(split_idx, len(ds)))

    def extract_ctx(r: dict) -> str:
        ctx = r.get("context") or {}
        if isinstance(ctx, dict) and "contexts" in ctx:
            return " ".join(ctx["contexts"])[:1200]
        return str(ctx)[:1200]

    def fmt(r: dict) -> str:
        ctx = extract_ctx(r)
        ans = f"{r['final_decision'].strip().upper()} — {r['long_answer'].strip()}"
        return (
            f"### Medical Abstract:\n{ctx}\n\n"
            f"### Question:\n{r['question'].strip()}\n\n"
            f"### Answer:\n{ans}<|endoftext|>\n\n"
        )

    train_texts = [fmt(r) for r in train_rows]
    val_texts = [fmt(r) for r in val_rows]
    eval_samples = [
        {
            "id": i,
            "question": r["question"].strip(),
            "correct_answer": r["final_decision"].strip(),
            "distractors": ["yes", "no", "maybe"],
            "support": extract_ctx(r),
        }
        for i, r in enumerate(val_rows)
        if i < 100
    ]
    rag_docs = list({extract_ctx(r) for r in train_rows if extract_ctx(r)})
    return _save_domain(
        BASE_DATA_DIR / "healthcare",
        "qiaojin/PubMedQA",
        "Healthcare (PubMed Biomedical Research Q&A)",
        train_texts,
        val_texts,
        eval_samples,
        rag_docs,
    )


def prepare_legal() -> dict:
    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    label_names = ds["train"].features["label"].names
    train_rows = ds["train"].select(range(6000))
    val_rows = ds["validation"].select(range(800))

    def fmt(r: dict) -> str:
        clause = r["text"].strip()[:1000]
        lbl = label_names[r["label"]]
        return (
            f"### Contract Clause:\n{clause}\n\n"
            f"### Question:\nWhat type of legal contract provision is this?\n\n"
            f"### Answer:\n{lbl}<|endoftext|>\n\n"
        )

    train_texts = [fmt(r) for r in train_rows]
    val_texts = [fmt(r) for r in val_rows]
    eval_samples = [
        {
            "id": i,
            "question": "What type of legal contract provision is this?",
            "correct_answer": label_names[r["label"]],
            "distractors": [],
            "support": r["text"].strip()[:1000],
        }
        for i, r in enumerate(val_rows)
        if i < 100
    ]
    rag_docs = [r["text"].strip()[:1000] for r in train_rows]
    return _save_domain(
        BASE_DATA_DIR / "legal",
        "coastalcph/lex_glue (ledgar)",
        "Legal (SEC Material Contract Clauses)",
        train_texts,
        val_texts,
        eval_samples,
        rag_docs,
    )


BUILDERS = {
    "sciq": prepare_sciq,
    "finance": prepare_finance,
    "healthcare": prepare_healthcare,
    "legal": prepare_legal,
}


def _extract_generic_record(row: dict, label_names: list[str] | None = None) -> tuple[str, str, str]:
    """Extract (context, question, answer) from diverse Hugging Face dataset schemas."""
    ctx = ""
    for k in ("context", "passage", "support", "document", "text", "summary", "input"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            ctx = v.strip()[:1200]
            break
        if isinstance(v, dict) and "contexts" in v:
            ctx = " ".join(v["contexts"])[:1200]
            break

    q = ""
    for k in ("question", "instruction", "prompt", "query", "title", "question_stem"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            q = v.strip()[:600]
            break

    ans = ""
    for k in ("answer", "response", "output", "best_answer", "long_answer", "code"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            ans = v.strip()[:800]
            break

    # Handle MMLU / ARC / OpenBookQA / MedMCQA multiple choice
    if not ans and "choices" in row:
        choices = row["choices"]
        c_texts = choices.get("text", []) if isinstance(choices, dict) else choices
        if isinstance(c_texts, list) and c_texts:
            q = q + "\nChoices: " + " | ".join(str(c) for c in c_texts[:6])
            ak = row.get("answerKey", row.get("answer", 0))
            if isinstance(ak, int) and 0 <= ak < len(c_texts):
                ans = str(c_texts[ak])
            elif isinstance(ak, str) and len(ak) == 1 and ak.upper() in "ABCD":
                idx = "ABCD".index(ak.upper())
                ans = str(c_texts[idx]) if idx < len(c_texts) else ak
    elif not ans and "cop" in row:  # MedMCQA
        opts = [row.get(k, "") for k in ("opa", "opb", "opc", "opd")]
        q = q + "\nChoices: " + " | ".join(str(o) for o in opts if o)
        cop = int(row["cop"]) if row["cop"] is not None else 0
        ans = str(opts[cop]) if 0 <= cop < len(opts) else ""

    # Handle classification datasets
    if not ans and "label" in row:
        lbl = row["label"]
        if isinstance(lbl, list):
            ans = ", ".join(label_names[i] if label_names and i < len(label_names) else str(i) for i in lbl) or "none"
        elif isinstance(lbl, int) and label_names and 0 <= lbl < len(label_names):
            ans = label_names[lbl]
        else:
            ans = str(lbl)

    if not q:
        q = "Analyze or classify the following domain text."
    if not ans:
        ans = "I don't know based on the provided context."
    return ctx, q, ans


def prepare_catalog_domain(domain_id: str, max_train: int = 4000, max_val: int = 400) -> dict:
    if domain_id in BUILDERS:
        return BUILDERS[domain_id]()
    if domain_id not in DOMAIN_CATALOG:
        raise ValueError(f"Unknown domain: {domain_id}")

    spec = DOMAIN_CATALOG[domain_id]
    ds = load_dataset(spec["repo"], spec["config"]) if spec["config"] else load_dataset(spec["repo"])
    splits = list(ds.keys())
    train_split = "train" if "train" in splits else splits[0]
    val_split = "validation" if "validation" in splits else ("test" if "test" in splits else None)

    if val_split and val_split != train_split:
        tr_ds = ds[train_split].select(range(min(max_train, len(ds[train_split]))))
        va_ds = ds[val_split].select(range(min(max_val, len(ds[val_split]))))
    else:
        full = ds[train_split].select(range(min(max_train + max_val, len(ds[train_split]))))
        cut = max(1, int(len(full) * 0.85))
        tr_ds = full.select(range(cut))
        va_ds = full.select(range(cut, len(full)))

    feat = tr_ds.features.get("label")
    label_names = getattr(feat, "names", None) or getattr(getattr(feat, "feature", None), "names", None)

    def fmt_row(row: dict) -> tuple[str, dict]:
        ctx, q, ans = _extract_generic_record(row, label_names)
        text = (
            (f"### Context:\n{ctx}\n\n" if ctx else "")
            + f"### Question:\n{q}\n\n### Answer:\n{ans}<|endoftext|>\n\n"
        )
        return text, {"question": q, "correct_answer": ans, "distractors": [], "support": ctx}

    train_texts, rag_docs = [], []
    seen_ctx: set[str] = set()
    for r in tr_ds:
        txt, rec = fmt_row(r)
        train_texts.append(txt)
        if rec["support"] and rec["support"] not in seen_ctx:
            seen_ctx.add(rec["support"])
            rag_docs.append(rec["support"])

    val_texts, eval_samples = [], []
    for i, r in enumerate(va_ds):
        txt, rec = fmt_row(r)
        val_texts.append(txt)
        if i < 100:
            eval_samples.append({"id": i, **rec})

    if not rag_docs:
        rag_docs = [s["question"] + " " + s["correct_answer"] for s in eval_samples[:50]]

    return _save_domain(
        BASE_DATA_DIR / domain_id,
        spec["repo"] + (f" ({spec['config']})" if spec["config"] else ""),
        f"{spec['category']} — {spec['label']}",
        train_texts,
        val_texts,
        eval_samples,
        rag_docs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare domain datasets from Hugging Face.")
    parser.add_argument(
        "--domain",
        choices=[*BUILDERS.keys(), *DOMAIN_CATALOG.keys(), "all"],
        default="all",
    )
    args = parser.parse_args()

    targets = list(BUILDERS.keys()) if args.domain == "all" else [args.domain]
    for name in targets:
        info = prepare_catalog_domain(name)
        print(f"[{name}] {info['domain']}: {info['train_tokens']:,} train tokens | {info['val_tokens']:,} val tokens")
