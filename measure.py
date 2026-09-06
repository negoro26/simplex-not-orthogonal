"""Geometry measurement: same-word angle histogram + weighted interference.

Pre-registered decision rule on median cross-sense cosine (after per-word
centering, k=4 senses -> simplex prediction cos = -1/3):
  median < -0.15              -> SIMPLEX-NEGATIVE (theory-optimal packing;
                                 per-word question dissolves, capacity
                                 relocates to cross-word interference)
  |median| <= 0.15            -> ORTHOGONAL CLUSTERS (regime-a-like)
  median > 0.15               -> COLLAPSED (check anisotropy artifact:
                                 must survive centering + patching to count)
Label-shuffle null reported alongside as the sampling-noise floor.
"""
import random

import torch


def centroids(caps_word):
    """caps_word: sense -> [vec]; returns dict sense -> normalized mean."""
    out = {}
    for s, vs in caps_word.items():
        mu = torch.stack(vs).mean(0)
        out[s] = mu / mu.norm()
    return out


def cross_cos(caps_word, rng=None, shuffle=False):
    senses = sorted(caps_word)
    if shuffle:
        pool = [v for s in senses for v in caps_word[s]]
        lab = pool[:]
        rng.shuffle(lab)
        reshuffled, i = {}, 0
        for s in senses:
            n = len(caps_word[s])
            reshuffled[s] = lab[i:i + n]
            i += n
        caps_word = reshuffled
    mus = centroids(caps_word)
    vals = []
    for i, a in enumerate(senses):
        for b in senses[i + 1:]:
            vals.append((mus[a] * mus[b]).sum().item())
    return vals


def analyze_word(w, caps, stream="resid"):
    cw = caps[stream][w]
    if sum(len(v) for v in cw.values()) < 8 or len(cw) < 2:
        return None
    base = torch.stack([torch.stack(vs).mean(0) for vs in cw.values()]).mean(0)
    centered = {s: [v - base for v in vs] for s, vs in cw.items()}
    return centered


def per_rank_stats(caps, stream="resid", seed=0):
    """Frequency-axis readout (the Scherlis-curve measurement).

    pair[(r1, r2)]: mean same-word cross-sense cos for that rank pair.
    rel[r]: split-half centroid self-consistency per rank = the noise
    ceiling; separation is only meaningful above it. Rare ranks have few
    test occurrences, so rel[r] is the honest denominator."""
    rng = random.Random(seed)
    pair, rel = {}, {}
    for w in sorted(caps[stream]):
        c = analyze_word(w, caps, stream)
        if not c or sum(len(v) for v in c.values()) < 8:
            continue
        mus = centroids(c)
        ranks = sorted(mus)
        for i, r1 in enumerate(ranks):
            vs = c[r1]
            if len(vs) >= 4:
                perm = list(vs)
                rng.shuffle(perm)
                h = len(perm) // 2
                m1 = torch.stack(perm[:h]).mean(0)
                m2 = torch.stack(perm[h:]).mean(0)
                rel.setdefault(r1, []).append(
                    (m1 / m1.norm() * m2 / m2.norm()).sum().item())
            for r2 in ranks[i + 1:]:
                pair.setdefault((r1, r2), []).append(
                    (mus[r1] * mus[r2]).sum().item())
    return ({k: sum(v) / len(v) for k, v in pair.items()},
            {k: sum(v) / len(v) for k, v in rel.items()},
            {k: len(v) for k, v in pair.items()})


def run_measurements(caps, meta, test_senses):
    from collections import Counter

    report = {}
    rng = random.Random(0)
    for stream in ("emb", "attn_raw", "resid"):
        real, null = [], []
        for w in sorted(caps[stream]):
            centered = analyze_word(w, caps, stream)
            if centered is None:
                continue
            real += cross_cos(centered)
            null += cross_cos(centered, rng=rng, shuffle=True)
        real_t = torch.tensor(real) if real else torch.tensor([float("nan")])
        null_t = torch.tensor(null) if null else torch.tensor([float("nan")])
        med = real_t.median().item()
        cls = ("SIMPLEX" if med < -0.15 else
               "COLLAPSED" if med > 0.15 else "ORTHOGONAL")
        bins = torch.histc(real_t, bins=20, min=-1, max=1)
        bar_max = max(bins.max().item(), 1)
        hist = "\n".join(
            f"    {(-1 + 0.1 * i):+.1f}..{(-0.9 + 0.1 * i):+.1f} {'#' * int(50 * b / bar_max)}"
            for i, b in enumerate(bins) if b > 0)
        report[stream] = dict(median=med, null_median=null_t.median().item(),
                              n_pairs=len(real), outcome=cls, hist=hist)

    # co-activation-weighted cross-word interference on the resid stream
    cents = {}
    for w in sorted(caps["resid"]):
        centered = analyze_word(w, caps, "resid")
        if centered:
            cents.update({(w, s): mu for s, mu in centroids(centered).items()})
    pair_count = Counter()
    for senses_in in test_senses:
        ss = sorted(set(senses_in))
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                if ss[i][0] != ss[j][0]:
                    pair_count[(ss[i], ss[j])] += 1
    total = sum(pair_count.values()) or 1
    interf = wsum = 0.0
    for (si, sj), c in pair_count.items():
        if si in cents and sj in cents:
            cos2 = (cents[si] * cents[sj]).sum().item() ** 2
            interf += cos2 * c / total
            wsum += c / total
    report["interference_weighted"] = interf
    report["interference_mass_covered"] = wsum
    report["rank_pair"], report["rank_rel"], report["rank_n"] = \
        per_rank_stats(caps)
    return report
