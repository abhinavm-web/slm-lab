"""Visual SLM Training & Hallucination Lab Server (FastAPI).

Serves a single-page educational UI where users can:
1. Explore tokenization & domain datasets (allenai/sciq, TinyShakespeare)
2. Configure model architecture & watch live GPU training on RTX 4060
3. Test generation with token-by-token confidence heatmaps
4. Experiment with Closed-Book vs. Grounded (RAG) hallucination behavior

Usage:
    python ui_server.py
"""

import importlib
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

model_mod = importlib.import_module("02_model")
GPT = model_mod.GPT
GPTConfig = model_mod.GPTConfig

eval_mod = importlib.import_module("08_eval_suite")
evaluate_checkpoint = eval_mod.evaluate_checkpoint

rag_mod = importlib.import_module("09_rag_retriever")
load_or_build_index = rag_mod.load_or_build_index
RAG_INDEXES: dict = {}

prep_mod = importlib.import_module("06_prepare_hf_dataset")
prepare_catalog_domain = prep_mod.prepare_catalog_domain
from domain_catalog import DOMAIN_CATALOG

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CKPT_DIR = BASE_DIR / "checkpoints"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="SLM Learning & Hallucination Lab")

TRAIN_LOCK = threading.Lock()
TRAIN_STATE: dict = {
    "status": "idle",  # idle | running | completed | stopped | error
    "message": "Ready to train.",
    "dataset": "sciq",
    "ckpt_name": "sciq_ckpt.pt",
    "step": 0,
    "max_iters": 400,
    "best_val_loss": None,
    "tok_per_sec": 0,
    "history": [],
    "error": None,
    "stop_requested": False,
}


class TokenizeReq(BaseModel):
    text: str


class EstimateReq(BaseModel):
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    vocab_size: int = 50257


class TrainStartReq(BaseModel):
    dataset: str = "sciq"
    ckpt_name: str = "sciq_ckpt.pt"
    n_layer: int = Field(default=6, ge=1, le=24)
    n_head: int = Field(default=6, ge=1, le=24)
    n_embd: int = Field(default=384, ge=32, le=2048)
    block_size: int = Field(default=256, ge=32, le=1024)
    batch_size: int = Field(default=16, ge=1, le=128)
    grad_accum_steps: int = Field(default=4, ge=1, le=32)
    max_iters: int = Field(default=400, ge=10, le=10000)
    learning_rate: float = Field(default=6e-4, gt=0, le=0.1)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    eval_interval: int = Field(default=50, ge=5, le=1000)


class GenerateReq(BaseModel):
    ckpt_name: str = "sciq_ckpt.pt"
    prompt: str
    max_tokens: int = 80
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95


class HallucinationReq(BaseModel):
    ckpt_name: str = "sciq_ckpt.pt"
    question: str
    support: str = ""
    correct_answer: str = ""
    temperature: float = 0.2
    max_tokens: int = 45
    expect_abstain: bool = False


class EvalReq(BaseModel):
    ckpt_name: str = "sciq_abstain_dpo.pt"
    compare_ckpt: str = ""
    dataset: str = "sciq"
    n_samples: int = Field(default=25, ge=5, le=100)


class RagReq(BaseModel):
    ckpt_name: str = "sciq_ckpt.pt"
    dataset: str = "sciq"
    query: str
    top_k: int = Field(default=3, ge=1, le=5)
    temperature: float = 0.2


def _get_dataset_dir(name: str) -> Path:
    if name == "shakespeare":
        return DATA_DIR
    candidate = DATA_DIR / name
    if candidate.exists() or name in DOMAIN_CATALOG:
        return candidate
    raise HTTPException(status_code=404, detail=f"Unknown dataset: {name}")


def _encode_text(text: str, meta: dict) -> list[int]:
    if meta.get("tokenizer_type", "gpt2") == "gpt2":
        enc = tiktoken.get_encoding("gpt2")
        return enc.encode(text, allowed_special={"<|endoftext|>"})
    stoi = meta["stoi"]
    return [stoi[c] for c in text if c in stoi]


def _decode_token(tok_id: int, meta: dict) -> str:
    if meta.get("tokenizer_type", "gpt2") == "gpt2":
        enc = tiktoken.get_encoding("gpt2")
        return enc.decode([tok_id])
    itos = {int(k): v for k, v in meta["itos"].items()}
    return itos.get(tok_id, "")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/system")
def system_info() -> dict:
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU Only"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2) if cuda_ok else 0

    datasets = []
    for ds_dir in sorted(DATA_DIR.iterdir()):
        if ds_dir.is_dir() and (ds_dir / "meta.json").exists():
            m = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))
            datasets.append({"id": ds_dir.name, "label": f"{ds_dir.name.upper()} — {m.get('domain', ds_dir.name)}"})
    if (DATA_DIR / "meta.json").exists():
        datasets.append({"id": "shakespeare", "label": "TinyShakespeare"})

    prepared_ids = {d["id"] for d in datasets}
    catalog = [
        {"id": k, "category": v["category"], "label": v["label"], "prepared": k in prepared_ids}
        for k, v in DOMAIN_CATALOG.items()
    ]

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpts = sorted([p.name for p in CKPT_DIR.glob("*.pt")])

    return {
        "cuda_available": cuda_ok,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "torch_version": torch.__version__,
        "datasets": datasets,
        "catalog": catalog,
        "checkpoints": ckpts,
    }


@app.post("/api/tokenize")
def tokenize_text(req: TokenizeReq) -> dict:
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(req.text, allowed_special={"<|endoftext|>"})
    tokens = [{"id": tid, "text": enc.decode([tid])} for tid in ids]
    chars_per_tok = round(len(req.text) / max(1, len(ids)), 2)
    return {
        "token_count": len(ids),
        "char_count": len(req.text),
        "chars_per_token": chars_per_tok,
        "tokens": tokens,
    }


class PrepareReq(BaseModel):
    domain_id: str


@app.post("/api/dataset/prepare")
def prepare_dataset_endpoint(req: PrepareReq) -> dict:
    try:
        info = prepare_catalog_domain(req.domain_id)
        return {"ok": True, "meta": info}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/dataset/{name}")
def dataset_info(name: str) -> dict:
    d = _get_dataset_dir(name)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not prepared yet.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    samples = []
    eval_path = d / "eval_samples.json"
    if eval_path.exists():
        samples = json.loads(eval_path.read_text(encoding="utf-8"))[:30]
    return {"meta": meta, "eval_samples": samples}


@app.post("/api/estimate")
def estimate_model(req: EstimateReq) -> dict:
    if req.n_embd % req.n_head != 0:
        raise HTTPException(status_code=400, detail="n_embd must be divisible by n_head")
    cfg = GPTConfig(
        vocab_size=req.vocab_size,
        block_size=req.block_size,
        n_layer=req.n_layer,
        n_head=req.n_head,
        n_embd=req.n_embd,
    )
    m = GPT(cfg)
    total = m.count_parameters(non_embedding=False)
    body = m.count_parameters(non_embedding=True)
    emb = total - body
    static_vram_mb = round((total * 16) / (1024 ** 2), 1)
    return {
        "total_params": total,
        "total_params_m": round(total / 1e6, 2),
        "embedding_params_m": round(emb / 1e6, 2),
        "block_params_m": round(body / 1e6, 2),
        "est_static_vram_mb": static_vram_mb,
    }


def _train_worker(req: TrainStartReq) -> None:
    global TRAIN_STATE
    try:
        d = _get_dataset_dir(req.dataset)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        train_data = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
        val_data = np.memmap(d / "val.bin", dtype=np.uint16, mode="r")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        gpt_cfg = GPTConfig(
            vocab_size=meta["vocab_size"],
            block_size=req.block_size,
            n_layer=req.n_layer,
            n_head=req.n_head,
            n_embd=req.n_embd,
            dropout=req.dropout,
            bias=False,
        )
        model = GPT(gpt_cfg).to(device)

        decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": 0.1},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=req.learning_rate,
            betas=(0.9, 0.95),
        )

        pt_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        ctx = torch.amp.autocast(device_type=device, dtype=pt_dtype, enabled=(device == "cuda"))

        def get_batch(data: np.memmap) -> tuple[torch.Tensor, torch.Tensor]:
            ix = torch.randint(len(data) - req.block_size - 1, (req.batch_size,))
            x = torch.stack([torch.from_numpy(data[i : i + req.block_size].astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + req.block_size].astype(np.int64)) for i in ix])
            return x.to(device), y.to(device)

        def get_lr(it: int) -> float:
            warmup = min(40, max(5, req.max_iters // 10))
            min_lr = req.learning_rate * 0.1
            if it < warmup:
                return req.learning_rate * (it + 1) / warmup
            ratio = (it - warmup) / max(1, req.max_iters - warmup)
            return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (req.learning_rate - min_lr)

        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        best_val = float("inf")
        t0 = time.perf_counter()
        tokens_per_step = req.batch_size * req.grad_accum_steps * req.block_size

        for it in range(req.max_iters + 1):
            if TRAIN_STATE["stop_requested"]:
                TRAIN_STATE["status"] = "stopped"
                TRAIN_STATE["message"] = f"Stopped by user at step {it}."
                break

            if it % req.eval_interval == 0 or it == req.max_iters:
                model.eval()
                eval_losses = {}
                with torch.no_grad():
                    for split, arr in [("train", train_data), ("val", val_data)]:
                        ls = []
                        for _ in range(15):
                            xb, yb = get_batch(arr)
                            with ctx:
                                _, loss = model(xb, yb)
                            ls.append(loss.item())
                        eval_losses[split] = float(np.mean(ls))
                model.train()

                elapsed = time.perf_counter() - t0
                vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
                ppl = math.exp(min(eval_losses["val"], 20.0))
                tok_s = int((it * tokens_per_step) / max(0.1, elapsed)) if it > 0 else 0

                if eval_losses["val"] < best_val:
                    best_val = eval_losses["val"]
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "model_config": gpt_cfg.__dict__,
                            "meta": meta,
                            "step": it,
                            "best_val_loss": best_val,
                        },
                        CKPT_DIR / req.ckpt_name,
                    )

                with TRAIN_LOCK:
                    TRAIN_STATE["step"] = it
                    TRAIN_STATE["best_val_loss"] = round(best_val, 3)
                    TRAIN_STATE["tok_per_sec"] = tok_s
                    TRAIN_STATE["history"].append(
                        {
                            "step": it,
                            "train_loss": round(eval_losses["train"], 3),
                            "val_loss": round(eval_losses["val"], 3),
                            "val_ppl": round(ppl, 1),
                            "vram_mb": round(vram_mb, 0),
                            "elapsed_s": round(elapsed, 1),
                        }
                    )

            if it == req.max_iters:
                with TRAIN_LOCK:
                    TRAIN_STATE["status"] = "completed"
                    TRAIN_STATE["message"] = f"Completed {req.max_iters} steps! Saved best checkpoint to {req.ckpt_name}."
                break

            lr = get_lr(it)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            for _ in range(req.grad_accum_steps):
                xb, yb = get_batch(train_data)
                with ctx:
                    _, loss = model(xb, yb)
                    loss = loss / req.grad_accum_steps
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with TRAIN_LOCK:
                TRAIN_STATE["step"] = it + 1

    except Exception as exc:
        with TRAIN_LOCK:
            TRAIN_STATE["status"] = "error"
            TRAIN_STATE["error"] = str(exc)
            TRAIN_STATE["message"] = f"Error: {exc}"


@app.post("/api/train/start")
def start_training(req: TrainStartReq) -> dict:
    if req.n_embd % req.n_head != 0:
        raise HTTPException(status_code=400, detail="Embedding width (n_embd) must be divisible by n_head.")
    with TRAIN_LOCK:
        if TRAIN_STATE["status"] == "running":
            raise HTTPException(status_code=409, detail="Training is already running.")
        TRAIN_STATE.update(
            {
                "status": "running",
                "message": f"Training {req.ckpt_name} on {req.dataset}...",
                "dataset": req.dataset,
                "ckpt_name": req.ckpt_name,
                "step": 0,
                "max_iters": req.max_iters,
                "best_val_loss": None,
                "tok_per_sec": 0,
                "history": [],
                "error": None,
                "stop_requested": False,
            }
        )
    threading.Thread(target=_train_worker, args=(req,), daemon=True).start()
    return {"ok": True}


@app.post("/api/train/stop")
def stop_training() -> dict:
    with TRAIN_LOCK:
        TRAIN_STATE["stop_requested"] = True
    return {"ok": True}


@app.get("/api/train/status")
def train_status() -> dict:
    with TRAIN_LOCK:
        return dict(TRAIN_STATE)


def _generate_with_confidence(
    ckpt_name: str, prompt: str, max_tokens: int, temperature: float, top_k: int, top_p: float
) -> dict:
    ckpt_path = CKPT_DIR / ckpt_name
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {ckpt_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gpt_cfg = GPTConfig(**ckpt["model_config"])
    meta = ckpt["meta"]
    eot_id = meta.get("eot_token", 50256)

    model = GPT(gpt_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt_ids = _encode_text(prompt, meta) or [0]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx[:, -gpt_cfg.block_size :]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]

            probs_raw = F.softmax(logits, dim=-1)

            if temperature <= 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                scaled = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(scaled, dim=-1)
                if 0.0 < top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True)
                    cp = torch.cumsum(sp, dim=-1)
                    sp[cp - sp > top_p] = 0.0
                    sp /= sp.sum(dim=-1, keepdim=True)
                    sampled = torch.multinomial(sp, num_samples=1)
                    idx_next = torch.gather(si, -1, sampled)
                else:
                    idx_next = torch.multinomial(probs, num_samples=1)

            tok_id = int(idx_next.item())
            if tok_id == eot_id:
                break

            tok_prob = float(probs_raw[0, tok_id].item())
            top3_p, top3_i = torch.topk(probs_raw[0], 3)
            alts = [
                {"token": _decode_token(int(i), meta), "prob": round(float(p) * 100, 1)}
                for p, i in zip(top3_p.tolist(), top3_i.tolist())
            ]
            generated_tokens.append(
                {
                    "token": _decode_token(tok_id, meta),
                    "prob_pct": round(tok_prob * 100, 1),
                    "alternatives": alts,
                }
            )
            idx = torch.cat((idx, idx_next), dim=1)

    text = "".join(t["token"] for t in generated_tokens)
    avg_conf = round(float(np.mean([t["prob_pct"] for t in generated_tokens])) if generated_tokens else 0.0, 1)
    return {
        "text": text,
        "tokens": generated_tokens,
        "avg_confidence_pct": avg_conf,
        "ckpt_val_loss": round(float(ckpt.get("best_val_loss", 0.0)), 3),
    }


@app.post("/api/generate")
def generate(req: GenerateReq) -> dict:
    return _generate_with_confidence(
        req.ckpt_name, req.prompt, req.max_tokens, req.temperature, req.top_k, req.top_p
    )


@app.post("/api/hallucination_test")
def hallucination_test(req: HallucinationReq) -> dict:
    closed_prompt = f"### Question:\n{req.question.strip()}\n\n### Answer:\n"
    closed_res = _generate_with_confidence(
        req.ckpt_name, closed_prompt, req.max_tokens, req.temperature, 20, 0.95
    )

    grounded_res = None
    if req.support.strip():
        grounded_prompt = (
            f"### Science Context:\n{req.support.strip()}\n\n"
            f"### Question:\n{req.question.strip()}\n\n### Answer:\n"
        )
        grounded_res = _generate_with_confidence(
            req.ckpt_name, grounded_prompt, req.max_tokens, req.temperature, 20, 0.95
        )

    def diagnose(res: dict | None) -> str:
        if res is None:
            return "N/A"
        ans = req.correct_answer.strip().lower()
        pred = res["text"].strip().lower()
        conf = res["avg_confidence_pct"]
        abstained = "don't know" in pred or "do not know" in pred or "not enough information" in pred

        if req.expect_abstain or ans in ("unknown", "i don't know", "n/a"):
            if abstained:
                return "Honest Abstention (Refused to Hallucinate!)"
            if conf >= 35.0:
                return "Confident Hallucination (invented an answer instead of abstaining)"
            return "Hallucinated Guess (did not abstain)"

        if ans and ans in pred:
            return "Grounded / Factual Match"
        if abstained:
            return "Over-Refusal (abstained even though answer was available)"
        if conf >= 35.0:
            return "Confident Hallucination (sounds sure, but missed ground truth)"
        return "Uncertain Guess (low token confidence)"

    return {
        "closed_book": {**closed_res, "diagnosis": diagnose(closed_res)},
        "grounded": ({**grounded_res, "diagnosis": diagnose(grounded_res)} if grounded_res else None),
    }


@app.post("/api/eval")
def run_eval(req: EvalReq) -> dict:
    primary = evaluate_checkpoint(CKPT_DIR / req.ckpt_name, n_samples=req.n_samples, dataset=req.dataset)
    baseline = None
    if req.compare_ckpt and req.compare_ckpt != req.ckpt_name:
        baseline = evaluate_checkpoint(CKPT_DIR / req.compare_ckpt, n_samples=req.n_samples, dataset=req.dataset)
    return {"primary": primary, "baseline": baseline}


@app.post("/api/rag")
def run_rag(req: RagReq) -> dict:
    if req.dataset not in RAG_INDEXES:
        RAG_INDEXES[req.dataset] = load_or_build_index(req.dataset)
    idx = RAG_INDEXES[req.dataset]
    hits = idx.search(req.query, top_k=req.top_k)
    top_ctx = hits[0]["text"] if hits else ""
    ctx_hdr = {
        "finance": "### Context:",
        "healthcare": "### Medical Abstract:",
        "legal": "### Contract Clause:",
    }.get(req.dataset, "### Science Context:")
    prompt = f"{ctx_hdr}\n{top_ctx}\n\n### Question:\n{req.query.strip()}\n\n### Answer:\n"
    gen = _generate_with_confidence(req.ckpt_name, prompt, 45, req.temperature, 20, 0.95)
    return {"hits": hits, "generation": gen, "indexed_docs": idx.n_docs}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)
