"""Cross-word packing vs capacity: the arm-5 epsilon-grid analysis.

Input: per-word-centered NORMALIZED sense centroids (the same objects the
interference measure uses; raw cosines are invalid per protocol). Output:
the cross-word |cos| distribution, violation curve on a log-spaced
±4°-10° grid, empirical crossover (median angular deviation from 90°),
and a matched random-direction baseline. No bound constant pre-committed:
the SGD-vs-random gap is the reported quantity.
"""
import math

import torch


def cross_word_abs_cos(cents, device="cpu", chunk=2048):
    """cents: {(word, rank): unit vec} -> 1-D tensor of cross-word |cos|."""
    keys = sorted(cents)
    M = torch.stack([cents[k] for k in keys]).to(device)
    wid = torch.tensor([k[0] for k in keys], device=device)
    out = []
    n = len(keys)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        g = (M[i0:i1] @ M.T).abs()
        cross = wid[i0:i1, None] != wid[None, :]
        # strict upper triangle of the full matrix, chunk-local rows
        cols = torch.arange(n, device=device)[None, :]
        upper = cols > (torch.arange(i0, i1, device=device)[:, None])
        out.append(g[cross & upper].cpu())
    return torch.cat(out)


def packing_report(cents, d, seed=0, device="cpu"):
    ac = cross_word_abs_cos(cents, device=device)
    n = len(cents)
    g = torch.Generator().manual_seed(seed)
    R = torch.nn.functional.normalize(
        torch.randn(n, d, generator=g), dim=-1).to(device)
    rand = {(i, 0): R[i].cpu() for i in range(n)}  # all distinct "words"
    ar = cross_word_abs_cos(rand, device=device)

    eps_deg = torch.logspace(math.log10(4.0), math.log10(10.0), 8)
    rows = []
    for e in eps_deg.tolist():
        s = math.sin(math.radians(e))
        rows.append((e, (ac > s).float().mean().item(),
                     (ar > s).float().mean().item()))
    q = torch.tensor([0.5, 0.9, 0.99, 0.999])

    def deg(t):  # |cos| -> angular deviation from 90° in degrees
        s = t.sort().values
        idx = (q * (len(s) - 1)).long()
        return torch.rad2deg(torch.asin(s[idx]))

    return dict(n_senses=n, n_pairs=len(ac),
                sgd_dev_deg={f"p{int(p * 1000) / 10}": round(v, 3)
                             for p, v in zip(q.tolist(), deg(ac).tolist())},
                rand_dev_deg={f"p{int(p * 1000) / 10}": round(v, 3)
                              for p, v in zip(q.tolist(), deg(ar).tolist())},
                grid=[(round(e, 2), round(v, 4), round(vr, 4))
                      for e, v, vr in rows],
                sgd_max_dev=torch.rad2deg(torch.asin(ac.max())).item(),
                rand_max_dev=torch.rad2deg(torch.asin(ar.max())).item())


def print_report(rep):
    print(f"packing: {rep['n_senses']} senses, {rep['n_pairs']} cross-word pairs")
    print(f"  angular deviation from 90° — SGD:    {rep['sgd_dev_deg']} max {rep['sgd_max_dev']:.2f}°")
    print(f"  angular deviation from 90° — random: {rep['rand_dev_deg']} max {rep['rand_max_dev']:.2f}°")
    print("  eps-grid  viol(SGD)  viol(random)   [fraction of pairs beyond 90°±eps]")
    for e, v, vr in rep["grid"]:
        print(f"    ±{e:5.2f}°   {v:.4f}     {vr:.4f}")
