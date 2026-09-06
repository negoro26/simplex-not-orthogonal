"""Post-hoc decomposition of arm-5 cross-word packing by corpus covariates.

Usage: python posthoc.py seed=0 [gamma=0.0] [lam=0.0] [prec=fp32]
       [pen=coact]
Reads results/big_g{gamma}_l{lam}_s{seed}{sfx}.pt; rebuilds the
deterministic corpus for the covariates (co-occurrence, cue sets).
"""
import sys
from collections import defaultdict

import torch

import harness

BIG = dict(n_words=2000, k=5, n_blocks=500, cues_per_pool=4,
           n_fillers=80, sents_per_word=48)


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def quantiles(t, ps):
    s = t.sort().values
    return [s[int(p * (len(s) - 1))].item() for p in ps]


def decompose(cents, seed, **corpus):
    """Cross-word deviation split by corpus covariates. Returns a dict with
    per-population quantiles and the Jaccard dose-response bins."""
    keys = sorted(cents)
    X = torch.stack([cents[k] for k in keys]).to(DEV)
    idx = {k: i for i, k in enumerate(keys)}
    N = len(keys)
    _, _, _, dgen = harness.prepare(seed=seed, **{**BIG, **corpus})
    sets = [frozenset(dgen["cue_seen"][k]) for k in keys]

    cooc = torch.zeros(N, N, dtype=torch.bool)
    for sent in dgen["senses_of"]:
        t = torch.tensor(sorted({idx[s] for s in sent if s in idx}))
        cooc[t[:, None], t[None, :]] = True
    by_cue = defaultdict(list)
    for i, s in enumerate(sets):
        for c in s:
            by_cue[c].append(i)
    shared = torch.zeros(N, N, dtype=torch.bool)
    for members in by_cue.values():
        t = torch.tensor(members)
        shared[t[:, None], t[None, :]] = True

    dev = torch.rad2deg(torch.asin((X @ X.T).abs().clamp(max=1.0)))
    wr = torch.tensor([k[0] for k in keys], device=DEV)
    m = (wr[:, None] != wr[None, :]).triu(1)
    fv, sh, co = dev[m], shared.to(DEV)[m], cooc.to(DEV)[m]
    s, ns = fv[sh], fv[~sh]
    rk = torch.tensor([k[1] for k in keys], device=DEV)
    same_rank = (rk[:, None] == rk[None, :])[m]
    s_same, s_cross = fv[sh & same_rank], fv[sh & ~same_rank]

    bins = defaultdict(list)
    pairs = {(min(i, j), max(i, j))
             for m in by_cue.values() for i in m for j in m
             if i != j and keys[i][0] != keys[j][0]}
    pairs = sorted(pairs)
    devs = dev[torch.tensor([i for i, _ in pairs]), torch.tensor([j for _, j in pairs])].tolist()
    for (i, j), dij in zip(pairs, devs):
        jac = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
        bins[min(jac // 0.2501 * 0.25 + 0.25, 1.0)].append(dij)
    jbins = {}
    for b, v in sorted(bins.items()):
        v.sort()
        jbins[round(b, 2)] = dict(n=len(v), med=v[len(v) // 2],
                                  p90=v[int(len(v) * .9)])
    return dict(
        n_pairs=len(fv), n_cooc=int(co.sum()), n_shared=len(s),
        cooc_mean=fv[co].mean().item(), noncooc_mean=fv[~co].mean().item(),
        shared=dict(zip(("med", "p90", "p999", "max"), quantiles(s, (.5, .9, .999, 1.0))),
                    p21=(s > 21.1).float().mean().item()),
        disjoint=dict(zip(("med", "p90", "p999", "max"), quantiles(ns, (.5, .9, .999, 1.0))),
                      p21=(ns > 21.1).float().mean().item()),
        shared_same_rank=dict(n=len(s_same), med=quantiles(s_same, (.5,))[0] if len(s_same) else float("nan")),
        shared_cross_rank=dict(n=len(s_cross), med=quantiles(s_cross, (.5,))[0] if len(s_cross) else float("nan")),
        jaccard=jbins)


if __name__ == "__main__":
    kw = dict(kv.split("=") for kv in sys.argv[1:])
    seed = int(kw.pop("seed", 0))
    gamma = float(kw.pop("gamma", 0.0))
    lam = float(kw.pop("lam", 0.0))
    prec = kw.pop("prec", "fp32")
    pen = kw.pop("pen", "coact")
    assert not kw, f"unknown args {kw}"
    sfx = "" if prec == "fp32" else f"_{prec}"
    if pen != "coact":
        sfx += f"_{pen}"
    d = torch.load(f"results/big_g{gamma}_l{lam}_s{seed}{sfx}.pt",
                   weights_only=False)
    r = decompose(d["cents"], seed)
    print(f"== posthoc g={gamma} l={lam} s={seed}: {r['n_pairs']:,} cross-word pairs")
    print(f"co-occurring: {r['n_cooc']:,}; mean dev {r['cooc_mean']:.2f} vs "
          f"{r['noncooc_mean']:.2f} (non)")
    for name in ("shared", "disjoint"):
        p = r[name]
        print(f"  dev|{name:8s} median {p['med']:5.2f}  p90 {p['p90']:5.2f}  "
              f"p99.9 {p['p999']:5.2f}  max {p['max']:.1f}  P(dev>21) {p['p21']:.5f}")
    print("  Jaccard dose-response (bin upper edge: n, median, p90):")
    for b, v in r["jaccard"].items():
        print(f"    <={b:4.2f}: n={v['n']:6}  med {v['med']:5.1f}  p90 {v['p90']:5.1f}")
