"""Stage 2: Decoder-Only Transformer (GPT-style) Architecture from Scratch.

Implements the core building blocks of modern LLMs:
1. Token + Positional Embeddings
2. Pre-Norm Transformer Blocks:
   - Multi-Head Causal Self-Attention (Q, K, V + causal mask)
   - Feed-Forward MLP (4x expansion + GELU)
   - Residual connections
3. Weight-tied Language Modeling Head (predicts next token logits)

Run directly to inspect parameter counts and VRAM estimates:
    python 02_model.py
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (modern LLMs typically omit bias)."""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention.

    Each token projects into Query (what am I looking for?), Key (what do I contain?),
    and Value (what information do I pass along?). Causal masking ensures token t
    can only attend to tokens <= t (no peeking at the future).
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Single fused projection for Q, K, V is faster than 3 separate Linear layers
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch, Sequence Length, Embedding Dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # PyTorch's scaled_dot_product_attention dispatches to FlashAttention / SDPA kernels
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise Feed-Forward Network (where factual/pattern knowledge is largely stored)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """One Transformer layer: Pre-Norm Attention + Pre-Norm MLP with residual connections."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),  # Positional embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share input embedding matrix with output projection matrix.
        # Reduces parameters significantly and improves generalization on small models.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layer) per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.transformer.wpe.weight.numel()
            total -= self.transformer.wte.weight.numel()
        return total

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            # ignore_index=-1 lets us mask out prompt tokens during SFT post-training
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: only compute logits for the final token
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = 0.95,
    ) -> torch.Tensor:
        """Autoregressive decoding loop: predict 1 token, append, repeat."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)

                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum_probs - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    sampled = torch.multinomial(sorted_probs, num_samples=1)
                    idx_next = torch.gather(sorted_idx, -1, sampled)
                else:
                    idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    cfg = GPTConfig()
    model = GPT(cfg)
    total_params = model.count_parameters(non_embedding=False)
    body_params = model.count_parameters(non_embedding=True)
    emb_params = total_params - body_params

    # Rough AdamW training memory: ~16 bytes/param (bf16 weights + grads + fp32 master + m + v)
    static_mem_mb = (total_params * 16) / (1024 ** 2)
    print(f"Total parameters:       {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  - Embeddings (wte+wpe): {emb_params:,} ({emb_params / 1e6:.2f}M)")
    print(f"  - Transformer blocks:   {body_params:,} ({body_params / 1e6:.2f}M)")
    print(f"Est. static AdamW VRAM: ~{static_mem_mb:.1f} MB (+ activations during forward/backward)")
