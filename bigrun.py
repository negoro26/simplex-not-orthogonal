"""Arm 5: capacity-bound run. d=4096, S=10^4 senses (2000 words x 5 ranks).

  python bigrun.py seed=0 [gamma=0.0] [lam=0.0] [steps=6000] [bs=256] [device=cuda]

Design registered in DISCOVERY.md before first run. Geometry is captured on
the FULL corpus (registered deviation; test split leaves <1 vector for rare
senses); behavioral per-rank accuracy stays on the held-out 10% split.
Only the resid stream is captured.
"""
import sys
import time

import torch
import torch.nn.functional as F

import harness
from measure import analyze_word, centroids, cross_cos, per_rank_stats
from packing import packing_report, print_report

# TF32 on Blackwell: ~3-4x on the fp32 matmuls that dominate this run.
# Registered amendment in DISCOVERY.md: ~1e-3 relative rounding, far below
# the centroid noise floor (split-half reliability). Statistics-critical
# gram matrices in packing_report run on CPU = strict fp32, unaffected.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BIG = dict(n_words=2000, k=5, n_blocks=500, cues_per_pool=4,
           n_fillers=80, sents_per_word=48)

kw = {}
for kv in sys.argv[1:]:
    k, v = kv.split("=")
    kw[k] = v
seed = int(kw.pop("seed", 0))
gamma = float(kw.pop("gamma", 0.0))
lam = float(kw.pop("lam", 0.0))
steps = int(kw.pop("steps", 6000))
lr = float(kw.pop("lr", 1e-4))
bs = int(kw.pop("bs", 256))
device = kw.pop("device", "cuda" if torch.cuda.is_available() else "cpu")
ckpt = kw.pop("ckpt", None)  # load weights, skip training (analysis-only)
prec = kw.pop("prec", "fp32")  # bf16 = autocast train (Amendment 5, gated)
pen = kw.pop("pen", "coact")  # uniform = inverse-frequency pair pricing
d_model = int(kw.pop("d", 4096))  # width sweep (registered 2026-09-02)
BIG["n_blocks"] = int(kw.pop("blocks", 500))
BIG["pool_by"] = kw.pop("pool", "rank")  # group = rank-mixed cue pools
assert not kw, f"unknown args {kw}"
sfx = "" if prec == "fp32" else f"_{prec}"
if pen != "coact":
    sfx += f"_{pen}"
if d_model != 4096:
    sfx += f"_d{d_model}"
if BIG["pool_by"] != "rank" or BIG["n_blocks"] != 500:
    sfx += f"_{BIG['pool_by']}{BIG['n_blocks']}"
if lr != 1e-4:
    sfx += f"_lr{lr:g}"

if device == "cuda":
    free, total = torch.cuda.mem_get_info()
    # train: 317M params x 16B (fp32 + grad + AdamW states = 5.1GB floor)
    # + block activations — measured 5.97GB alloc / 6.12GB reserved;
    # analysis-only: params + gather transients — measured 4.0GB reserved.
    need = 4.4e9 if ckpt else 6.8e9
    assert free >= need, (
        f"preflight: {free / 1e9:.1f}GB VRAM free, need {need / 1e9:.1f}GB "
        f"— close GPU-hungry apps before burning an hour")
    print(f"preflight [{torch.cuda.get_device_name()}]: "
          f"{free / 1e9:.1f}GB VRAM free (need {need / 1e9:.1f})", flush=True)

train_data, test_data, meta, d_gen = harness.prepare(seed=seed, **BIG)
V = meta["V"]
print(f"== BIG d={d_model} gamma={gamma} lam={lam} seed={seed} V={V} "
      f"train={len(train_data)} test={len(test_data)} device={device} "
      f"prec={prec} pen={pen}", flush=True)

word_of = None
freq_of = None
if lam:
    word_of = torch.full((V,), -1, dtype=torch.long)
    for tid, (w, _r) in meta["cont_id"].items():
        word_of[tid] = w
    if pen == "uniform":
        total_main = sum(d_gen["main_count"].values())
        freq_of = torch.zeros(V)
        for tid, (w, r) in meta["cont_id"].items():
            freq_of[tid] = d_gen["main_count"][(w, r)] / total_main

ckpt_path = f"results/big_g{gamma}_l{lam}_s{seed}{sfx}.ckpt"
if ckpt:
    from model import TinyLM
    model = TinyLM(V, d_model)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)
    print(f"loaded {ckpt}, skipping training", flush=True)
else:
    model, secs = harness.train(V=V, data=train_data, d_model=d_model,
                                gamma=gamma, lam=lam, word_of=word_of,
                                steps=steps, bs=bs, lr=lr, seed=seed,
                                device=device, log_every=500, prec=prec,
                                pen=pen, freq_of=freq_of)
    torch.save(model.state_dict(), ckpt_path)
    print(f"train {secs:.0f}s  saved {ckpt_path}", flush=True)


@torch.no_grad()
def big_capture(rows, want_vecs, cap_bs=512):
    """One pass, in order, each row once. Returns (X, labels, by_rank)."""
    vecs, labels = [], []
    br = {}
    for off in range(0, len(rows), cap_bs):
        x, y, m = harness.pad_rows(rows[off:off + cap_bs])
        resid = model.features(x.to(device))
        sel = (m.to(device)[:, :-1] > 0)
        hs = resid[:, :-1][sel]
        tgt = y.to(device)[:, :-1][sel].long()
        logits = model.unemb(hs)
        hit = (logits.argmax(-1) == tgt)
        ce = F.cross_entropy(logits, tgt, reduction="none")
        for t, h, c in zip(tgt.tolist(), hit.tolist(), ce.tolist()):
            w, r = meta["cont_id"][t]
            s = br.setdefault(r, [0, 0.0, 0])
            s[0] += int(h); s[1] += c; s[2] += 1
            if want_vecs:
                labels.append((w, r))
        if want_vecs:
            vecs.append(hs.cpu())
    X = torch.cat(vecs) if want_vecs else None
    by_rank = {r: dict(acc=ok / n, loss=ls / n, n=n)
               for r, (ok, ls, n) in br.items()}
    return X, labels, by_rank

t0 = time.time()
_, _, test_rank = big_capture(test_data, want_vecs=False)
acc = sum(v["acc"] * v["n"] for v in test_rank.values()) / \
      sum(v["n"] for v in test_rank.values())
print(f"held-out cont acc {acc:.3f}")
print("per-rank (held-out):",
      {f"r{r}": f"acc={v['acc']:.3f} loss={v['loss']:.3f} n={v['n']}"
       for r, v in sorted(test_rank.items())})

X, labels, _ = big_capture(train_data + test_data, want_vecs=True)
print(f"capture {time.time() - t0:.0f}s  ({len(labels)} vectors)", flush=True)
if device == "cuda":
    torch.cuda.empty_cache()

caps = {"resid": {}}
for i, (w, r) in enumerate(labels):
    caps["resid"].setdefault(w, {}).setdefault(r, []).append(X[i])

# same-word geometry (pre-registered decision rule) + centered centroids
real, cents = [], {}
for w in sorted(caps["resid"]):
    centered = analyze_word(w, caps)
    if centered is None:
        continue
    real += cross_cos(centered)
    cents.update({(w, s): mu for s, mu in centroids(centered).items()})
med = torch.tensor(real).median().item()
cls = ("SIMPLEX" if med < -0.15 else
       "COLLAPSED" if med > 0.15 else "ORTHOGONAL")
print(f"[resid] same-word median cos {med:+.3f} (n={len(real)} pairs, "
      f"k=5 simplex=-0.25) -> {cls}")

pair, rel, _n = per_rank_stats(caps)
print("rank-pair mean cos:",
      {f"r{a}-r{b}": round(v, 3) for (a, b), v in sorted(pair.items())})
print("split-half reliability:",
      {f"r{r}": round(v, 3) for r, v in sorted(rel.items())})

# co-activation-weighted cross-word interference, full-corpus weights
from collections import Counter
pair_count = Counter()
for senses_in in d_gen["senses_of"]:
    ss = sorted(set(senses_in))
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            if ss[i][0] != ss[j][0]:
                pair_count[(ss[i], ss[j])] += 1
total = sum(pair_count.values())
interf = wsum = 0.0
for (si, sj), c in pair_count.items():
    if si in cents and sj in cents:
        interf += (cents[si] * cents[sj]).sum().item() ** 2 * c / total
        wsum += c / total
print(f"weighted cross-word interference: {interf:.4f} "
      f"(pair mass covered {wsum:.2f})")
del caps, X  # release the 1.6GB vector matrix before the gram matrices
# packing on CPU: strict fp32 gram, exempt from the TF32 amendment
rep = packing_report(cents, d=d_model, seed=seed, device="cpu")
print_report(rep)
out = f"results/big_g{gamma}_l{lam}_s{seed}{sfx}.pt"
torch.save(dict(cents=cents, packing=rep, test_rank=test_rank, acc=acc,
                med=med, outcome=cls, rank_pair=pair, rank_rel=rel,
                interference=interf, config=dict(BIG, steps=steps, bs=bs,
                gamma=gamma, lam=lam, seed=seed, prec=prec, pen=pen, d=d_model, lr=lr)), out)
print(f"saved {out}")
