"""Post-hoc checks raised in adversarial review (2026-09-02; NOT pre-registered).

Usage: python nulls.py seed=0 [gamma=0.0] [lam=0.0] [prec=fp32] [pen=coact]

R1  Centering identity. analyze_word subtracts the unweighted mean of the k
    sense centroids; for any k equal-norm equiangular vectors the centered
    pairwise cosine is exactly -1/(k-1) whatever the raw cosine. So report the
    RAW same-word geometry (no centering), global-mean-centered geometry, and
    the protocol (per-word-centered) value side by side, plus the ratio of the
    per-word base norm to the sense-deviation norm.
R2  d_eff explains the bulk? For isotropic unit vectors in d_eff dims the
    median |cos| ~ 0.6745/sqrt(d_eff). Report (a) that prediction at d_eff =
    PR of the centroids, (b) a spectrum-matched Gaussian null: N(0, Sigma) with
    Sigma the empirical centroid covariance, normalized, cross-word deviation
    quantiles, (c) the isotropic-in-R^d null used in the paper.
R3  Random-init baseline: the same pipeline on an untrained TinyLM (same seed):
    protocol same-word cosine, rank-pair matrix, cross-word deviation, PR.
"""
import math
import sys

import torch
import torch.nn.functional as F

import harness
from model import TinyLM
from posthoc import decompose

BIG = dict(n_words=2000, k=5, n_blocks=500, cues_per_pool=4,
           n_fillers=80, sents_per_word=48)
kw = dict(kv.split("=") for kv in sys.argv[1:])
seed = int(kw.pop("seed", 0))
gamma = float(kw.pop("gamma", 0.0))
lam = float(kw.pop("lam", 0.0))
prec = kw.pop("prec", "fp32")
pen = kw.pop("pen", "coact")
d_model = int(kw.pop("d", 4096))
BIG["n_blocks"] = int(kw.pop("blocks", 500))
BIG["pool_by"] = kw.pop("pool", "rank")
lr = float(kw.pop("lr", 1e-4))
assert not kw, kw
sfx = ("" if prec == "fp32" else f"_{prec}") + ("" if pen == "coact" else f"_{pen}")
if d_model != 4096:
    sfx += f"_d{d_model}"
if BIG["pool_by"] != "rank" or BIG["n_blocks"] != 500:
    sfx += f"_{BIG['pool_by']}{BIG['n_blocks']}"
if lr != 1e-4:
    sfx += f"_lr{lr:g}"
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
K = BIG["k"]

train_data, test_data, meta, _ = harness.prepare(seed=seed, **BIG)
V = meta["V"]


@torch.no_grad()
def capture(model, rows):
    hs, tg = [], []
    for off in range(0, len(rows), 512):
        x, y, m = harness.pad_rows(rows[off:off + 512])
        xd = x.to(dev)
        sel = m.to(dev)[:, :-1] > 0
        hs.append(model.features(xd)[:, :-1][sel])
        tg.append(y.to(dev)[:, :-1][sel].long())
    return torch.cat(hs), torch.cat(tg)


def sense_means(H, tgt):
    """(S, d) raw centroid matrix over complete words, ordered (word, rank)."""
    keys = sorted(meta["cont_id"].values())
    idx = {s: i for i, s in enumerate(keys)}
    lab = torch.tensor([idx[meta["cont_id"][int(t)]] for t in tgt.tolist()], device=dev)
    cnt = torch.bincount(lab, minlength=len(keys))
    M = torch.zeros(len(keys), H.shape[1], device=dev).index_add_(0, lab, H)
    M /= cnt.clamp(min=1)[:, None]
    word = torch.tensor([s[0] for s in keys], device=dev)
    rank = torch.tensor([s[1] for s in keys], device=dev)
    ok_word = (cnt.view(-1, K) > 0).all(1)          # words with all K senses seen
    keep = ok_word[word]
    print(f"  senses {len(keys)}, complete words {int(ok_word.sum())}")
    return M[keep], word[keep], rank[keep]


def same_word(M, word, rank, center):
    """Median cross-sense cosine and rank-pair means under a centering mode."""
    if center == "perword":
        Mw = M.view(-1, K, M.shape[1])            # keys sorted (word, rank) -> contiguous
        Mc = (Mw - Mw.mean(1, keepdim=True)).view_as(M)
    elif center == "global":
        Mc = M - M.mean(0)
    else:
        Mc = M
    U = F.normalize(Mc, dim=1).view(-1, K, M.shape[1])
    G = U @ U.transpose(1, 2)                     # (W, K, K)
    iu = torch.triu_indices(K, K, 1)
    pairs = G[:, iu[0], iu[1]]                    # (W, 10)
    return pairs.median().item(), {f"r{iu[0][j] + 1}-r{iu[1][j] + 1}": round(pairs[:, j].mean().item(), 3)
                                   for j in range(pairs.shape[1])}


def cross_dev(U, word):
    parts = []
    n = len(U)
    for i0 in range(0, n, 2000):
        G = (U[i0:i0 + 2000] @ U.T).abs().clamp(max=1)
        cols = torch.arange(n, device=dev)[None, :]
        m = (word[i0:i0 + 2000, None] != word[None, :]) & (cols > torch.arange(i0, min(i0 + 2000, n), device=dev)[:, None])
        parts.append(torch.rad2deg(torch.asin(G[m])))
    d = torch.cat(parts).sort().values
    q = lambda p: d[int(p * (len(d) - 1))].item()  # noqa: E731
    return q(0.5), q(0.999), d.max().item()


def pr(X):
    s2 = torch.linalg.svdvals(X - X.mean(0)) ** 2
    return (s2.sum() ** 2 / (s2 ** 2).sum()).item()


def protocol_centroids(M):
    Mw = M.view(-1, K, M.shape[1])
    return F.normalize((Mw - Mw.mean(1, keepdim=True)).view_as(M), dim=1)


def report(tag, M, word, rank):
    print(f"\n== {tag}")
    for mode in ("raw", "global", "perword"):
        med, rp = same_word(M, word, rank, mode)
        print(f"  same-word median cos [{mode:8s}] {med:+.3f}   rank pairs {rp}")
    Mw = M.view(-1, K, M.shape[1])
    base = Mw.mean(1, keepdim=True)
    ratio = (base.norm(dim=2).mean() / (Mw - base).norm(dim=2).mean()).item()
    print(f"  per-word base norm / sense-deviation norm = {ratio:.2f}")
    Mn = F.normalize(M, dim=1)
    print(f"  UNCENTERED sense centroids: PR {pr(Mn):.0f} (global mean removed), "
          f"cross-word dev p50/p99.9/max {'/'.join(f'{x:.1f}' for x in cross_dev(Mn, word))}")
    Wm = F.normalize(M.view(-1, K, M.shape[1]).mean(1), dim=1)
    Gw = (Wm @ Wm.T).abs()
    Gw.fill_diagonal_(0)
    print(f"  WORD centroids ({Wm.shape[0]} words): PR {pr(Wm):.0f}, cross-word |cos| median {Gw[Gw > 0].median():.3f}, p99.9 {Gw[Gw > 0].flatten().kthvalue(int(0.999 * (Gw.numel() - Wm.shape[0]))).values:.3f}")
    U = protocol_centroids(M)
    p50, p999, mx = cross_dev(U, word)
    P = pr(U)
    print(f"  cross-word dev (protocol centroids): p50 {p50:.2f}  p99.9 {p999:.1f}  max {mx:.1f}   PR {P:.0f}")
    print(f"  isotropic prediction at d_eff=PR: p50 ~ {math.degrees(math.asin(0.6745 / math.sqrt(P))):.2f}")
    g = torch.Generator(device=dev).manual_seed(seed)
    R = F.normalize(torch.randn(U.shape, generator=g, device=dev), dim=1)
    print(f"  null A, isotropic R^d:         p50 {cross_dev(R, torch.arange(len(U), device=dev))[0]:.2f}"
          f"  p99.9 {cross_dev(R, torch.arange(len(U), device=dev))[1]:.2f}")
    Uc = U - U.mean(0)
    L = torch.linalg.cholesky(Uc.T @ Uc / len(Uc) + 1e-6 * torch.eye(U.shape[1], device=dev))
    Z = F.normalize(torch.randn(U.shape, generator=g, device=dev) @ L.T + U.mean(0), dim=1)
    a, b, c = cross_dev(Z, word)
    print(f"  null B, spectrum-matched Gaussian (word structure kept): p50 {a:.2f}  p99.9 {b:.1f}  max {c:.1f}")
    cents = {(int(w), int(r)): U[i].cpu() for i, (w, r) in enumerate(zip(word.tolist(), rank.tolist()))}
    dc = decompose(cents, seed, n_blocks=BIG["n_blocks"], pool_by=BIG["pool_by"])
    print(f"  shared-cue pairs: med {dc['shared']['med']:.1f}  p90 {dc['shared']['p90']:.1f}  P(>21°) {dc['shared']['p21']:.3f}  n {dc['n_shared']:,}"
          f"   | disjoint: med {dc['disjoint']['med']:.2f}  P(>21°) {dc['disjoint']['p21']:.4f}")
    print(f"  shared-cue same-rank med {dc['shared_same_rank']['med']:.1f} (n {dc['shared_same_rank']['n']:,})"
          f"   cross-rank med {dc['shared_cross_rank']['med']:.1f} (n {dc['shared_cross_rank']['n']:,})")
    print("  Jaccard bins (upper edge: med, n): " + ", ".join(f"{b}: {v['med']:.1f} ({v['n']})" for b, v in dc["jaccard"].items()))


print(f"== nulls g={gamma} l={lam} s={seed}{sfx}")
torch.manual_seed(seed)
init = TinyLM(V, d_model).to(dev).eval()
H, tgt = capture(init, train_data)
Mwr = sense_means(H, tgt)
del H, init
torch.cuda.empty_cache()
report("RANDOM-INIT model (untrained, same seed)", *Mwr)

model = TinyLM(V, d_model)
model.load_state_dict(torch.load(f"results/big_g{gamma}_l{lam}_s{seed}{sfx}.ckpt",
                                 map_location="cpu", weights_only=False))
model = model.to(dev).eval()
H, tgt = capture(model, train_data)
Mwr = sense_means(H, tgt)
del H, model
torch.cuda.empty_cache()
report("TRAINED checkpoint", *Mwr)
