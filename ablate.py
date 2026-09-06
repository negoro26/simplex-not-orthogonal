"""Causal ablation of the shared-cue subspace at the residual readout.

Usage: python ablate.py seed=0 [gamma=0.0] [lam=0.0] [prec=fp32] [pen=coact]
       [ks=0,1,2,4,8,16,32]

Reads results/big_g{gamma}_l{lam}_s{seed}{sfx}.ckpt; pure analysis, no
training. The shared-cue subspace is built from TRAIN-side data only: for
every cross-word sense pair sharing a cue pool (same group, same rank —
by construction the pairs with |cos| ~ 0.8), the normalized pair mean is
one sample of their COMMON component. PCA of those samples gives an
orthonormal basis; held-out behavior is evaluated with the projection
onto its top-k directions subtracted from every residual vector before
the unembedding (readout ablation, training untouched).

Controls at matched k:
  random   seeded random orthonormal subspace
  generic  top-k PCA of all train residual vectors (variance, not pairing)

Pre-registered predictions (2026-08-24, before first execution):
  P1  shared ablation degrades rare-rank held-out behavior (r5/r4)
      monotonically in k; common ranks r1/r2 barely move.
  P2  matched-k controls degrade all ranks roughly uniformly, and hit r5
      less than the shared subspace does.
  P3  same-word median cos (train-side estimate, registered protocol)
      largely survives shared ablation — the damage is behavioral,
      through lost context information, not a destroyed simplex.
Falsifying P1+P2 weakens "load-bearing" to mere persistence; confirming
both localizes WHAT the aligned population computes.

Check #1 — deletion-fairness controls (registered 2026-09-02, before first
execution; rule text below). Objection: the shared-cue subspace may simply
be "the directions that carry rank/context information", which any
rank-informative basis would delete just as destructively. New bases, all
from TRAIN rows only:
  rank     orthonormal span of the 5 per-rank mean residuals (k <= 5 by
           construction; rows for k > 5 are skipped, not padded)
  cue      top-k PCA of per-cue-token mean residuals (cue = x[t-1] at the
           prediction position; the "context-hint part")
  rare     top-k PCA of r4/r5 sense centroids ("hard-case" directions)
  randvar  random orthonormal subspace whose column count k' is the
           smallest that removes at least as much train-residual variance
           as `shared` at nominal k (fairness on deleted energy, not on
           dimension count)
Per row we also report the fraction of held-out residual variance removed
and the median principal-angle cosine between the basis and `shared` at the
same k. Decision rule:
  PASS     at matched k and matched removed variance, shared's r5/r4 damage
           exceeds every control by more than the gamma=0 seed spread, and
           median cos(shared, rank|cue) at k=16 < 0.7  -> ablation stays
           the centerpiece.
  FAIL     rank or cue reproduce the shared damage curve within seed noise
           -> stop; do not run Check #2; reframe as "SGD builds a shared
           context channel instead of an orthogonal code".
  PARTIAL  shared beats rank/cue on r5 but overlaps them (cos >= 0.7)
           -> "low-dimensional carrier of context information"; drop
           "dedicated channel".
"""
import sys

import torch
import torch.nn.functional as F

import harness
from measure import analyze_word, cross_cos

BIG = dict(n_words=2000, k=5, n_blocks=500, cues_per_pool=4,
           n_fillers=80, sents_per_word=48)

kw = dict(kv.split("=") for kv in sys.argv[1:])
seed = int(kw.pop("seed", 0))
gamma = float(kw.pop("gamma", 0.0))
lam = float(kw.pop("lam", 0.0))
prec = kw.pop("prec", "fp32")
pen = kw.pop("pen", "coact")
BIG["n_blocks"] = int(kw.pop("blocks", 500))
BIG["pool_by"] = kw.pop("pool", "rank")
ks = [int(x) for x in kw.pop("ks", "0,1,2,4,8,16,32").split(",")]
assert not kw, f"unknown args {kw}"
sfx = "" if prec == "fp32" else f"_{prec}"
if pen != "coact":
    sfx += f"_{pen}"
if BIG["pool_by"] != "rank" or BIG["n_blocks"] != 500:
    sfx += f"_{BIG['pool_by']}{BIG['n_blocks']}"

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"

from model import TinyLM  # noqa: E402

train_data, test_data, meta, dgen = harness.prepare(seed=seed, **BIG)
V = meta["V"]
model = TinyLM(V, 4096)
model.load_state_dict(torch.load(
    f"results/big_g{gamma}_l{lam}_s{seed}{sfx}.ckpt",
    map_location="cpu", weights_only=False))
model = model.to(dev).eval()


@torch.no_grad()
def capture(rows):
    """Masked-position residual rows on GPU + their target and cue token
    ids. Sentence layout is `<bos> cue w cont filler ...`, so at a masked
    position t the cue is x[t-1]."""
    hs_all, tgts, cues = [], [], []
    for off in range(0, len(rows), 512):
        x, y, m = harness.pad_rows(rows[off:off + 512])
        xd = x.to(dev)
        sel = (m.to(dev)[:, :-1] > 0)
        resid = model.features(xd)
        hs_all.append(resid[:, :-1][sel])
        tgts.append(y.to(dev)[:, :-1][sel].long())
        cues.append(xd.roll(1, dims=1)[:, :-1][sel])
    return torch.cat(hs_all), torch.cat(tgts), torch.cat(cues)


print(f"== ablate g={gamma} l={lam} s={seed}{sfx}: "
      f"train={len(train_data)} test={len(test_data)}", flush=True)
Htr, tgt_tr, cue_tr = capture(train_data)
Hte, tgt_te, _ = capture(test_data)
labels = [meta["cont_id"][int(t)] for t in tgt_te.cpu()]
ranks_te = torch.tensor([r for _w, r in labels], device=dev)

# ---- sense centroids from train rows --------------------------------------
sums, counts = {}, {}
for row, t in zip(Htr.cpu(), tgt_tr.cpu()):
    s = meta["cont_id"][int(t)]
    if s not in sums:
        sums[s] = torch.zeros(4096)
        counts[s] = 0
    sums[s] += row
    counts[s] += 1
cents = {s: F.normalize(v / counts[s], dim=0) for s, v in sums.items()}

# ---- shared-cue pair means -> covariance -> eigenbasis ---------------------
# Pools are read from the corpus (senses sharing a cue token), so this works for
# rank-indexed pools (standard corpus) and rank-mixed pools (pool=group) alike.
n_blocks = BIG["n_blocks"]
k_ranks = BIG["k"]
senses_by_gr = {}
for s, cues_ in dgen["cue_seen"].items():
    if s in cents and cues_:
        pool = next(iter(cues_)).rsplit("_", 1)[0]     # cue-name prefix = pool id
        senses_by_gr.setdefault(pool, []).append(s)

gen = torch.Generator().manual_seed(seed)
M_batches = []
for members in senses_by_gr.values():
    if len(members) < 2:
        continue
    idx = torch.randperm(len(members), generator=gen)
    for i in range(len(idx) // 2):  # disjoint pairs
        a, b = members[idx[2 * i]], members[idx[2 * i + 1]]
        M_batches.append(F.normalize(cents[a] + cents[b], dim=0))
M = torch.stack(M_batches).to(dev)
C = (M.T @ M) / M.shape[0]
evals, evecs = torch.linalg.eigh(C)
U_shared = evecs.flip(1)                     # descending variance order
var_shared = evals.flip(0) / evals.sum()

Hc = Htr - Htr.mean(0)
Cg = (Hc.T @ Hc) / Hc.shape[0]
evg, evecg = torch.linalg.eigh(Cg)
U_generic = evecg.flip(1)
Gq, _ = torch.linalg.qr(torch.randn(4096, max(ks), generator=gen))
U_rand = Gq.to(dev)

# ---- Check #1 fairness bases (train rows only) -----------------------------
def pca_basis(rows):
    """Descending-variance orthonormal basis of centered `rows`."""
    Xc = rows - rows.mean(0)
    ev, evec = torch.linalg.eigh((Xc.T @ Xc) / Xc.shape[0])
    return evec.flip(1)


ranks_tr = torch.tensor([meta["cont_id"][int(t)][1] for t in tgt_tr.cpu()],
                        device=dev)
rank_means = torch.stack([Htr[ranks_tr == r].mean(0)
                          for r in range(1, k_ranks + 1)])
U_rank = torch.linalg.svd(rank_means.T, full_matrices=False)[0]  # k <= 5

cue_sum = torch.zeros(V, 4096, device=dev).index_add_(0, cue_tr, Htr)
cue_n = torch.bincount(cue_tr, minlength=V)
cue_means = cue_sum[cue_n > 0] / cue_n[cue_n > 0, None]
U_cue = pca_basis(cue_means)
del cue_sum, cue_means

rare_cents = torch.stack([v for s, v in cents.items() if s[1] >= k_ranks - 1]).to(dev)
U_rare = pca_basis(rare_cents)
del rare_cents

Rfull, _ = torch.linalg.qr(torch.randn(4096, 4096, generator=gen))
Rfull = Rfull.to(dev)
tr_energy = float((Htr ** 2).sum())
col_energy = torch.zeros(4096, device=dev)
for off in range(0, Htr.shape[0], 8192):
    col_energy += ((Htr[off:off + 8192] @ Rfull) ** 2).sum(0)
rand_cum = torch.cumsum(col_energy, 0) / tr_energy


def removed_var(H, Uk):
    return float(((H @ Uk) ** 2).sum() / (H ** 2).sum())


def randvar_basis(k):
    """Smallest random-column prefix removing >= shared-k's train variance."""
    target = removed_var(Htr, basis(U_shared, k))
    kp = int(torch.searchsorted(rand_cum, torch.tensor(target, device=dev))) + 1
    return Rfull[:, :min(kp, 4096)].contiguous()


def med_pa_cos(Ua, Ub):
    """Median principal-angle cosine between two orthonormal bases."""
    return float(torch.linalg.svdvals(Ua.T @ Ub).median())


del Hc, Cg, C
torch.cuda.empty_cache()
print(f"shared directions: {M.shape[0]} pair means; top-8 var share "
      f"{[round(v, 3) for v in var_shared[:8].tolist()]}", flush=True)
print(f"cue means: {int((cue_n > 0).sum())} cue tokens; rank basis dims "
      f"{U_rank.shape[1]}", flush=True)


def basis(U, k):
    return U[:, :k].contiguous()


@torch.no_grad()
def evaluate(H):
    logits = model.unemb(H)
    hit = (logits.argmax(-1) == tgt_te)
    ce = F.cross_entropy(logits, tgt_te, reduction="none")
    out = {}
    masks = [("all", torch.ones_like(ranks_te, dtype=torch.bool))]
    masks += [(f"r{r}", ranks_te == r) for r in range(1, BIG["k"] + 1)]
    for tag, mask in masks:
        out[tag] = (float(hit[mask].float().mean()), float(ce[mask].mean()),
                    int(mask.sum()))
    return out


def med_cos_train(k, U):
    Hb = Htr - (Htr @ basis(U, k)) @ basis(U, k).T if k else Htr
    caps = {"resid": {}}
    Hb_cpu = Hb.cpu()
    del Hb
    lab_tr = [meta["cont_id"][int(t)] for t in tgt_tr.cpu()]
    for i, (w, r) in enumerate(lab_tr):
        caps["resid"].setdefault(w, {}).setdefault(r, []).append(Hb_cpu[i])
    vals = []
    for w in sorted(caps["resid"]):
        centered = analyze_word(w, caps)
        if centered is not None:
            vals += cross_cos(centered)
    return float(torch.tensor(vals).median())


hdr = (f"{'variant':>8} {'k':>4} {'acc':>6} "
       + " ".join(f"r{r}" for r in range(1, BIG["k"] + 1))
       + "   r5loss  medcos  var%  cos·sh")
print(hdr, flush=True)
for name, U in (("base", None), ("shared", U_shared),
                ("generic", U_generic), ("random", U_rand),
                ("rank", U_rank), ("cue", U_cue), ("rare", U_rare),
                ("randvar", None)):
    for k in ks:
        mc = float("nan")
        pa = float("nan")
        kshow = k
        if name == "base":
            if k != 0:
                continue
            Hb, vr = Hte, 0.0
        else:
            if k == 0 or (name == "rank" and k > U_rank.shape[1]):
                continue
            Uk = randvar_basis(k) if name == "randvar" else basis(U, k)
            kshow = Uk.shape[1]
            Hb = Hte - (Hte @ Uk) @ Uk.T
            vr = removed_var(Hte, Uk)
            if name != "shared":
                pa = med_pa_cos(basis(U_shared, k), Uk)
        res = evaluate(Hb)
        if name in ("base", "shared") and k in (0, 16):
            mc = med_cos_train(k, U if name == "shared" else None)
            torch.cuda.empty_cache()
        print(f"{name:>8} {kshow:>4} {res['all'][0]:.4f} "
              + " ".join(f"{res[f'r{r}'][0]:.3f}"
                         for r in range(1, BIG["k"] + 1))
              + f"  {res['r5'][1]:.3f}  {mc:+.3f}  {100 * vr:4.1f}  {pa:.3f}",
              flush=True)
        del Hb
