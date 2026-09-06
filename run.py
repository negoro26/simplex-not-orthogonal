"""One run:  python3 run.py d=64 gamma=0.01 seed=0 [steps=1500] [grouping=blocks]"""
import sys

import harness
from measure import run_measurements

kw = {}
for kv in sys.argv[1:]:
    k, v = kv.split("=")
    kw[k] = v
for k in ("d", "seed", "steps", "sents_per_word"):
    if k in kw:
        kw[k] = int(kw[k])
for k in ("gamma", "lam"):
    if k in kw:
        kw[k] = float(kw[k])
d_model = kw.pop("d", 64)
seed = kw.get("seed", 0)

train_data, test_data, meta, _ = harness.prepare(
    seed=seed, sents_per_word=kw.pop("sents_per_word", 1600),
    grouping=kw.pop("grouping", "blocks"))
print(f"== d={d_model} gamma={kw.get('gamma', 0.0)} lam={kw.get('lam', 0.0)} "
      f"seed={seed} V={meta['V']} train={len(train_data)} test={len(test_data)}",
      flush=True)

if kw.get("lam"):
    import torch
    word_of = torch.full((meta["V"],), -1, dtype=torch.long)
    for tid, (w, _r) in meta["cont_id"].items():
        word_of[tid] = w
    kw["word_of"] = word_of

model, secs = harness.train(V=meta["V"], data=train_data,
                            d_model=d_model, **kw)
caps, acc, by_rank = harness.capture(model, test_data, meta)
rep = run_measurements(caps, meta, meta["test_senses"])

print(f"cont acc {acc:.3f}  ({secs:.0f}s)")
print("per-rank behavior:",
      {f"r{r}": f"acc={v['acc']:.3f} loss={v['loss']:.3f} n={v['n']}"
       for r, v in sorted(by_rank.items())})
for stream in ("emb", "attn_raw", "resid"):
    r = rep[stream]
    print(f"[{stream:>8}] median cos {r['median']:+.3f}  "
          f"(null {r['null_median']:+.3f}, n={r['n_pairs']}) -> {r['outcome']}")
    print(r["hist"])
print(f"weighted cross-word interference (resid): "
      f"{rep['interference_weighted']:.4f}  "
      f"(pair mass covered {rep['interference_mass_covered']:.2f})")
pm, rel = rep["rank_pair"], rep["rank_rel"]
print("same-word rank-pair mean cos:",
      {f"r{a}-r{b}": round(v, 3) for (a, b), v in sorted(pm.items())})
print("split-half reliability by rank:",
      {f"r{r}": round(v, 3) for r, v in sorted(rel.items())})
