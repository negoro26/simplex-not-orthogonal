"""Minimal from-scratch transformer for the sense-allocation study.

Registered choices:
- no LayerNorm anywhere: global AdamW weight decay is exactly the
  Liu et al. superposition knob (LN would break its scale-invariance
  semantics and turn gamma into an effective-LR reparametrization)
- untied embedding/unembedding: token-identity geometry cannot leak
  into the unembedding space and contaminate sense measurements
- single-head attention; ReLU MLP (toy-model lineage); no biases
- no positional embeddings: chunk order carries no signal by construction
- forward exposes attn_raw (attention output PRE-out_proj/PRE-residual)
  and the final residual, so the readout point is explicit, never hooked
  through library fast paths

Pad skipping (CUDA only; engineering, not a registered knob): rows are
right-padded with PAD, attention is causal and there is no positional code,
so a pad position can never reach a real token and no reader consumes pad
outputs. Every per-token GEMM (qkv, out, MLP) therefore runs on the real
tokens only; the tiny T x T attention runs on the padded layout with pad
q/k/v = 0. Pad outputs come back as exact zeros. Corpus lengths are
8/11/14 with batches padded to 14, so this removes ~21% of block compute.
CPU (toy) runs keep the dense path.
"""
import torch
import torch.nn as nn

PAD = 0  # token id 0 is reserved for padding by harness.prepare (stoi = i+1)


class Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, keep=None):
        B, T, D = x.shape
        if keep is None:
            qkv = self.qkv(x)
        else:
            qr = self.qkv(x[keep])
            qkv = qr.new_zeros(B, T, 3 * D)
            qkv[keep] = qr
        q, k, v = qkv.split(D, dim=-1)
        att = q @ k.transpose(-1, -2) / (D ** 0.5)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        att = att.masked_fill(~causal, float("-inf"))
        a = att.softmax(-1) @ v
        return self.out(a if keep is None else a[keep]), a


class Block(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        self.attn = Attn(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mult * d, bias=False),
            nn.ReLU(),
            nn.Linear(mult * d, d, bias=False),
        )

    def forward(self, x, keep=None):
        proj, raw = self.attn(x, keep)
        h = (x if keep is None else x[keep]) + proj
        r = h + self.mlp(h)
        if keep is not None:
            full = r.new_zeros(x.shape)
            full[keep] = r
            r = full
        return r, raw


class TinyLM(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.block = Block(d)
        self.unemb = nn.Linear(d, vocab, bias=False)  # untied
        for p in self.parameters():
            nn.init.normal_(p, std=0.02)

    @staticmethod
    def _keep(idx):
        return (idx != PAD) if idx.is_cuda else None

    def features(self, idx):
        """Post-block residual stream without the unembedding, so the
        trainer can unembed only the loss-bearing positions instead of
        materializing B*T*V logits. Same tensor as streams['resid']."""
        return self.block(self.emb(idx), self._keep(idx))[0]

    def forward(self, idx):
        e = self.emb(idx)
        resid, attn_raw = self.block(e, self._keep(idx))
        return self.unemb(resid), {"attn_raw": attn_raw, "emb": e, "resid": resid}
