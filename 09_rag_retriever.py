"""Stage 9: Lightweight Vector RAG Retriever over Science Passages.

Builds a TF-IDF vector index over the allenai/sciq textbook passages (zero extra
dependencies) so users can ask any science question, inspect the Top-K retrieved
documents + similarity scores, and feed the best context to the SLM.

Usage:
    python 09_rag_retriever.py --query "How do plants make food from sunlight?"
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "data" / "sciq" / "rag_docs.json"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class SimpleRAGIndex:
    def __init__(self, docs: list[str]) -> None:
        self.docs = docs
        self.n_docs = len(docs)
        self.df: Counter[str] = Counter()
        self.doc_tfs: list[Counter[str]] = []

        for d in docs:
            toks = _tokenize(d)
            tf = Counter(toks)
            self.doc_tfs.append(tf)
            for term in tf:
                self.df[term] += 1

        self.idf: dict[str, float] = {
            term: math.log((self.n_docs + 1) / (freq + 1)) + 1.0
            for term, freq in self.df.items()
        }
        self.doc_norms: list[float] = []
        for tf in self.doc_tfs:
            sq = sum((cnt * self.idf[t]) ** 2 for t, cnt in tf.items())
            self.doc_norms.append(math.sqrt(sq) if sq > 0 else 1.0)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q_tf = Counter(_tokenize(query))
        q_vec = {t: cnt * self.idf.get(t, 0.0) for t, cnt in q_tf.items() if t in self.idf}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0:
            return [{"rank": i + 1, "score": 0.0, "text": self.docs[i]} for i in range(min(top_k, self.n_docs))]

        scores: list[tuple[float, int]] = []
        for idx, tf in enumerate(self.doc_tfs):
            dot = sum(q_val * tf[t] * self.idf[t] for t, q_val in q_vec.items() if t in tf)
            if dot > 0:
                sim = dot / (q_norm * self.doc_norms[idx])
                scores.append((sim, idx))

        scores.sort(reverse=True)
        return [
            {"rank": r + 1, "score": round(sim, 3), "text": self.docs[idx]}
            for r, (sim, idx) in enumerate(scores[:top_k])
        ]


def load_or_build_index(dataset: str = "sciq") -> SimpleRAGIndex:
    idx_path = BASE_DIR / "data" / dataset / "rag_docs.json"
    if idx_path.exists():
        docs = json.loads(idx_path.read_text(encoding="utf-8"))
        return SimpleRAGIndex(docs)

    ds = load_dataset("allenai/sciq", split="train")
    seen: set[str] = set()
    docs: list[str] = []
    for r in ds:
        sup = (r.get("support") or "").strip()
        if sup and sup not in seen:
            seen.add(sup)
            docs.append(sup)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps(docs), encoding="utf-8")
    return SimpleRAGIndex(docs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG retrieval over Science passages.")
    parser.add_argument("--query", type=str, default="What gas do plants absorb during photosynthesis?")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    idx = load_or_build_index()
    print(f"[rag] Indexed {idx.n_docs:,} unique science passages.")
    for hit in idx.search(args.query, top_k=args.top_k):
        print(f"  #{hit['rank']} (score={hit['score']}): {hit['text'][:160]}...")
