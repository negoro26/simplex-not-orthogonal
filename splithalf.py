"""Split-half signal PR (registered 2026-09-05, DISCOVERY.md "Why does PR grow
with width?"). For each sense, split its training sentences into two random
halves A, B and form centroids. Noise is independent across halves, so
    Σ_signal = ½(Cov(A,B) + Cov(B,A))   is unbiased for the signal covariance,
    Σ_noise  = ½ Cov(A − B)             is the per-centroid noise covariance,
and PR(Σ) = tr(Σ)² / tr(Σ²). Reported for the protocol (per-word-centered) and
the uncentered sense code, with and without r5.

Usage: python splithalf.py seed=0 [prec=bf16] [d=4096] [lr=1e-4] [blocks=500] [pool=rank]
"""
import sys

import torch

import harness
from model import TinyLM

BIG = dict(n_words=2000, k=5, n_blocks=500, cues_per_pool=4,
           n_fillers=80, sents_per_word=48)
kw = dict(kv.split("=") for kv in sys.argv[1:])
seed = int(kw.pop("seed", 0))
prec = kw.pop("prec", "fp32")
d_model = int(kw.pop("d", 4096))
lr = float(kw.pop("lr", 1e-4))
BIG["n_blocks"] = int(kw.pop("blocks", 500))
BIG["pool_by"] = kw.pop("pool", "rank")
assert not kw, kw
sfx = "" if prec == "fp32" else f"_{prec}"
if d_model != 4096:
    sfx += f"_d{d_model}"
if BIG["pool_by"] != "rank" or BIG["n_blocks"] != 500:
    sfx += f"_{BIG['pool_by']}{BIG['n_blocks']}"
if lr != 1e-4:
    sfx += f"_lr{lr:g}"
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
K = BIG["k"]

train_data, _, meta, _ = harness.prepare(seed=seed, **BIG)
model = TinyLM(meta["V"], d_model)
model.load_state_dict(torch.load(f"results/big_g0.0_l0.0_s{seed}{sfx}.ckpt", map_location="cpu", weights_only=False))
model = model.to(dev).eval()

hs, tg = [], []
with torch.no_grad():
    for off in range(0, len(train_data), 512):
        x, y, m = harness.pad_rows(train_data[off:off + 512])
        _, aux = model(x.to(dev))
        sel = m.to(dev)[:, :-1] > 0
        hs.append(aux["resid"][:, :-1][sel].float())
        tg.append(y.to(dev)[:, :-1][sel].long())
H, tgt = torch.cat(hs), torch.cat(tg)
del model, hs, tg

keys = sorted(meta["cont_id"].values())
idx = {s: i for i, s in enumerate(keys)}
lab = torch.tensor([idx[meta["cont_id"][int(t)]] for t in tgt.tolist()], device=dev)
rank = torch.tensor([s[1] for s in keys], device=dev)
g = torch.Generator(device=dev).manual_seed(seed + 77)
half = torch.rand(len(lab), generator=g, device=dev) < 0.5


def centroids(mask):
    cnt = torch.bincount(lab[mask], minlength=len(keys))
    M = torch.zeros(len(keys), H.shape[1], device=dev).index_add_(0, lab[mask], H[mask])
    return M / cnt.clamp(min=1)[:, None], cnt


A, ca = centroids(half)
B, cb = centroids(~half)
ok = ((ca.view(-1, K) > 0) & (cb.view(-1, K) > 0)).all(1)     # words with every sense in both halves
keep = ok.repeat_interleave(K)
print(f"== split-half s={seed}{sfx}: {int(ok.sum())} complete words, "
      f"sentences per sense median {int(torch.cat([ca, cb])[keep.repeat(2)].float().median())} per half")


def protocol(M):
    Mw = M.view(-1, K, M.shape[1])
    return (Mw - Mw.mean(1, keepdim=True)).view_as(M)


def pr(S):
    return (torch.trace(S) ** 2 / (S * S).sum()).item()


def report(name, A_, B_):
    Ac, Bc = A_ - A_.mean(0), B_ - B_.mean(0)
    n = len(Ac)
    Sab = Ac.T @ Bc / n
    S_signal = 0.5 * (Sab + Sab.T)
    D = (Ac - Bc)
    S_noise = 0.5 * (D.T @ D / n)
    S_raw = 0.5 * (Ac.T @ Ac + Bc.T @ Bc) / n
    print(f"  {name:32s} PR raw {pr(S_raw):6.1f}   PR signal {pr(S_signal):6.1f}   PR noise {pr(S_noise):7.1f}   "
          f"signal share of trace {torch.trace(S_signal).item() / torch.trace(S_raw).item():.2f}")


for label, m in (("all ranks", keep), ("r1-r4 only", keep & (rank != K))):
    print(f" [{label}]")
    report("protocol (per-word-centered)", protocol(A)[m], protocol(B)[m])
    report("uncentered sense centroids", A[m], B[m])
    Aw, Bw = A.view(-1, K, A.shape[1]).mean(1)[ok], B.view(-1, K, B.shape[1]).mean(1)[ok]
    if label == "all ranks":
        report("word centroids", Aw, Bw)
