"""Targeted probe: per-rank separation in healthy models + gamma window hunt.

Runs (sequential, solo -- oomd discipline):
  d=64 g=0.0 s=0, s=1   anchors with full per-rank readout
  d=64 g=1.0 s=0, g=2.0 s=0   intermediate-decay window probe
"""
import harness
from measure import run_measurements

for d_model, g, seed in ((64, 0.0, 0), (64, 0.0, 1), (64, 1.0, 0), (64, 2.0, 0)):
    tr, te, meta, _ = harness.prepare(seed=seed)
    model, secs = harness.train(V=meta["V"], data=tr, d_model=d_model,
                                gamma=g, steps=2500, seed=seed,
                                log_every=10 ** 9)
    caps, acc, by_rank = harness.capture(model, te, meta)
    rep = run_measurements(caps, meta, meta["test_senses"])
    r, a = rep["resid"], rep["attn_raw"]
    print(f"--- d={d_model} g={g} s={seed}: acc={acc:.3f} "
          f"resid_med={r['median']:+.3f} (null {r['null_median']:+.3f}) "
          f"attn_med={a['median']:+.3f} interf={rep['interference_weighted']:.3f}",
          flush=True)
    print("    per-rank:", {f"r{r}": f"acc={v['acc']:.3f} loss={v['loss']:.3f}"
                            for r, v in sorted(by_rank.items())})
    print("    rank-pair mean cos:",
          {f"r{p[0]}-r{p[1]}": round(v, 3) for p, v in sorted(rep["rank_pair"].items())})
    print("    split-half reliability:",
          {f"r{k}": round(v, 3) for k, v in sorted(rep["rank_rel"].items())})
