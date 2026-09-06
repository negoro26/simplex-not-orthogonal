"""Figure data for the paper (CSV for pgfplots). Usage: python figs.py
Writes paper/data/*.csv from saved artifacts; no training, no plotting deps."""
import glob
import math
import os
import re

import torch

from posthoc import BIG, decompose
import harness

torch.backends.cuda.matmul.allow_tf32 = False
os.makedirs("paper/data", exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Fig: deviation histograms, seed 0 --------------------------------------
a = torch.load("results/big_g0.0_l0.0_s0.pt", weights_only=False)
keys = sorted(a["cents"])
X = torch.stack([a["cents"][k] for k in keys]).to(dev)
wid = torch.tensor([k[0] for k in keys], device=dev)
_, _, _, dgen = harness.prepare(seed=0, **BIG)
by_cue = {}
for i, k in enumerate(keys):
    for c in dgen["cue_seen"][k]:
        by_cue.setdefault(c, []).append(i)
shared = torch.zeros(len(keys), len(keys), dtype=torch.bool, device=dev)
for m in by_cue.values():
    t = torch.tensor(m, device=dev)
    shared[t[:, None], t[None, :]] = True
G = (X @ X.T).abs().clamp(max=1.0)
cross = (wid[:, None] != wid[None, :]).triu(1)
devs = torch.rad2deg(torch.asin(G[cross]))
sh = shared[cross]
R = torch.nn.functional.normalize(torch.randn(len(keys), 4096, generator=torch.Generator().manual_seed(0)), dim=-1).to(dev)
rdev = torch.rad2deg(torch.asin((R @ R.T).abs().clamp(max=1.0).triu(1)[torch.ones_like(cross).triu(1)]))
edges = torch.logspace(math.log10(0.05), math.log10(90), 61, device=dev)
with open("paper/data/dev_hist.csv", "w") as f:
    f.write("lo,hi,sgd,shared,disjoint,random\n")
    for lo, hi in zip(edges[:-1].tolist(), edges[1:].tolist()):
        def frac(v):
            return ((v >= lo) & (v < hi)).float().mean().item() / (hi - lo)
        f.write(f"{lo:.4f},{hi:.4f},{frac(devs):.6g},{frac(devs[sh]):.6g},{frac(devs[~sh]):.6g},{frac(rdev):.6g}\n")
del G, R

# ---- Fig: ablation curves, gamma=0 seed 0 with Check #1 controls -------------
rows = []
for line in open("results/abl_g0.0_l0.0_s0_c1.log"):
    m = re.match(r"\s*(\w+)\s+(\d+)\s+([\d.]+)\s+([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)\s+([\d.]+)\s+\S+\s+([\d.]+)", line)
    if m:
        rows.append(m.groups())
with open("paper/data/ablation.csv", "w") as f:
    f.write("variant,k,acc,r1,r2,r3,r4,r5,r5loss,var\n")
    for r in rows:
        f.write(",".join(r) + "\n")

# ---- Fig: Jaccard dose-response across gamma=0 seeds ------------------------
with open("paper/data/jaccard.csv", "w") as f:
    f.write("seed,bin,n,med,p90\n")
    for p in sorted(glob.glob("results/big_g0.0_l0.0_s[0-9].pt")):
        d = torch.load(p, weights_only=False)
        r = decompose(d["cents"], d["config"]["seed"])
        for b, v in r["jaccard"].items():
            f.write(f"{d['config']['seed']},{b},{v['n']},{v['med']:.2f},{v['p90']:.2f}\n")

# ---- Table: rank-pair cosine matrices (toy gamma=0 mean of 5; capacity mean of 4)
import ast
toy = []
for p in glob.glob("results/grid_0.0_s*.log"):
    t = open(p).read()
    toy.append(ast.literal_eval(re.search(r"rank-pair mean cos: (\{.*\})", t).group(1)))
cap = [torch.load(p, weights_only=False)["rank_pair"] for p in sorted(glob.glob("results/big_g0.0_l0.0_s[0-9].pt"))]
with open("paper/data/rankpair.csv", "w") as f:
    f.write("arm,pair,mean,min,max\n")
    for name, runs in (("toy", toy), ("capacity", cap)):
        for pr in sorted(runs[0]):
            vals = [r[pr] for r in runs]
            key = pr if isinstance(pr, str) else f"r{pr[0]}-r{pr[1]}"
            f.write(f"{name},{key},{sum(vals) / len(vals):.3f},{min(vals):.3f},{max(vals):.3f}\n")
with open("paper/data/rankpair_matrix.csv", "w") as f:
    f.write("arm,i,j,cos\n")
    for name, runs_, kk in (("toy", toy, 4), ("capacity", cap, 5)):
        for i in range(1, kk + 1):
            for j in range(1, kk + 1):
                if i == j:
                    f.write(f"{name},{i},{j},nan\n")
                    continue
                a, b = min(i, j), max(i, j)
                key = f"r{a}-r{b}" if name == "toy" else (a, b)
                f.write(f"{name},{i},{j},{sum(r[key] for r in runs_) / len(runs_):.3f}\n")

# ---- Fig: eigenvalue spectra of centered sense centroids ----------------------
def spectrum(M, n=256):
    s2 = torch.linalg.svdvals(M.float()) ** 2
    return (s2[:n] / s2.sum()).tolist()


runs = {"gamma0": "results/big_g0.0_l0.0_s0.pt",
        "coact": "results/big_g0.0_l1.0_s0_bf16.pt",
        "uniform": "results/big_g0.0_l1.0_s0_bf16_uniform.pt"}
spec = {}
for name, p in runs.items():
    d = torch.load(p, weights_only=False)
    C = torch.stack([d["cents"][k] for k in sorted(d["cents"])]).to(dev)
    spec[name] = spectrum(C - C.mean(0))
G = torch.randn(10000, 4096, generator=torch.Generator().manual_seed(0)).to(dev)
spec["gaussian"] = spectrum(G - G.mean(0))
del G
with open("paper/data/spectrum.csv", "w") as f:
    f.write("i," + ",".join(spec) + "\n")
    for i in range(256):
        f.write(f"{i + 1}," + ",".join(f"{spec[n][i]:.6g}" for n in spec) + "\n")

# ---- Fig: principal-angle overlap (basis x k) to shared, mean over 5 ckpts ---
logs = ["results/abl_g0.0_l0.0_s0_c1.log"] + sorted(glob.glob("results/abl_c1_*.log"))
ov = {}
for p in logs:
    for line in open(p):
        m = re.match(r"\s*(generic|random|rank|cue|rare|randvar)\s+(\d+)\s.*\s([\d.]+)\s*$", line)
        if m:
            ov.setdefault((m.group(1), p), []).append(float(m.group(3)))
bases = ["rank", "cue", "generic", "rare", "randvar", "random"]
ks = [1, 2, 4, 8, 16, 32]
with open("paper/data/overlap.csv", "w") as f:
    f.write("basis,b,k,ki,cos\n")
    for bi, b in enumerate(bases):
        vals = [ov[(b, p)] for p in logs if (b, p) in ov]
        for ki, k in enumerate(ks):
            col = [v[ki] for v in vals if ki < len(v)]
            if col:
                f.write(f"{b},{bi},{k},{ki},{sum(col) / len(col):.3f}\n")

# ---- Fig: toy gamma dose-response (5 seeds) ----------------------------------
with open("paper/data/toy_gamma.csv", "w") as f:
    f.write("gamma,r34,r34lo,r34hi,oth,othlo,othhi,acc,acclo,acchi,interf\n")
    byg = {}
    for p in glob.glob("results/grid_*_s*.log"):
        t = open(p).read()
        g = float(re.search(r"grid_([\d.]+)_s", p).group(1))
        pair = ast.literal_eval(re.search(r"rank-pair mean cos: (\{.*\})", t).group(1))
        acc = float(re.search(r"cont acc ([\d.]+)", t).group(1))
        itf = float(re.search(r"interference \(resid\): ([\d.]+)", t).group(1))
        oth = sum(v for kk, v in pair.items() if kk != "r3-r4") / 5
        byg.setdefault(g, []).append((pair["r3-r4"], oth, acc, itf))
    for g, rows_ in sorted(byg.items()):
        r34 = sorted(r[0] for r in rows_)
        oth = sorted(r[1] for r in rows_)
        acc = sorted(r[2] for r in rows_)
        itf = sorted(r[3] for r in rows_)
        med = lambda v: v[len(v) // 2]  # noqa: E731
        f.write(f"{g},{med(r34):.3f},{r34[0]:.3f},{r34[-1]:.3f},{med(oth):.3f},{oth[0]:.3f},{oth[-1]:.3f},"
                f"{med(acc):.3f},{acc[0]:.3f},{acc[-1]:.3f},{med(itf):.3f}\n")
print("wrote paper/data/*.csv")

# ---- Fig: width sweep and rank-mixed pools, straight from TABLES.md T9 / T10 ---
def md_table(path, heading):
    sec = open(path).read().split(heading, 1)[1].split("\n## ", 1)[0]
    rows = [r for r in sec.splitlines() if r.startswith("|") and not r.startswith("|---")]
    return [[c.strip() for c in r.strip("|").split("|")] for r in rows]


hdr, *rows = md_table("results/TABLES.md", "## T9.")
# per (d, lr) seed-0 rows; best-of-lr accuracy per width; split-half signal PR from its log
by = {}
for r in rows:
    d, seed, lr = int(r[0]), int(r[1]), float(r[2])
    if seed == 0:
        by[(d, lr)] = dict(acc=float(r[3]), pr=float(r[7]))
sig = {}
for line in open("results/splithalf_s0.log"):
    if line.startswith("== split-half"):
        m = re.search(r"_d(\d+)", line)
        cur = int(m.group(1)) if m else 4096
    elif "protocol (per-word-centered)" in line and cur not in sig:
        sig[cur] = float(re.search(r"PR signal\s+([\d.]+)", line).group(1))
with open("paper/data/width.csv", "w") as f:
    f.write("d,acc_lr1e4,acc_lr1e3,acc_best,pr,pr_signal\n")
    for d in sorted({k[0] for k in by}):
        a4, a3 = by.get((d, 1e-4), {}).get("acc", float("nan")), by.get((d, 1e-3), {}).get("acc", float("nan"))
        cands = [v for v in (by.get((d, lr)) for lr in (1e-4, 1e-3, 3e-4)) if v]
        best = max(cands, key=lambda v: v["acc"])
        f.write(f"{d},{a4:.3f},{a3:.3f},{best['acc']:.3f},{best['pr']:.0f},{sig.get(d, float('nan')):.1f}\n")
hdr, *rows = md_table("results/TABLES.md", "## T10.")
with open("paper/data/mixed.csv", "w") as f:
    f.write("seed,acc,pr,p50,p999,max,nullB_p50,nullB_p999,nullB_max,shared_med,disjoint_med,p21_shared,p21_disjoint,n_shared\n")
    for r in rows:
        b50, b999, bmax = r[8].split(" / ")
        ps, pd = r[11].split(" / ")
        f.write(",".join([r[0], r[1], r[4], r[5], r[6], r[7], b50, b999, bmax, r[9], r[10], ps, pd, r[12].replace(",", "")]) + "\n")
