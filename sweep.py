"""gamma sweep: d x gamma x seed, one summary line per run."""
import harness
from measure import run_measurements
for d_model in (32, 64):
    for g in (0.0, 0.5, 5.0):
        for seed in (0, 1, 2):
            tr, te, meta, _ = harness.prepare(seed=seed)
            model, secs = harness.train(V=meta["V"], data=tr,
                                        d_model=d_model, gamma=g,
                                        steps=2500, seed=seed,
                                        log_every=10 ** 9)
            caps, acc, _ = harness.capture(model, te, meta)
            rep = run_measurements(caps, meta, meta["test_senses"])
            r = rep["resid"]
            a = rep["attn_raw"]
            print(f"d={d_model:2d} g={g:<5} s={seed} acc={acc:.3f} "
                  f"resid_med={r['median']:+.3f} null={r['null_median']:+.3f} "
                  f"[{r['outcome']:<9}] attn_med={a['median']:+.3f} "
                  f"interf={rep['interference_weighted']:.3f} ({secs:.0f}s)",
                  flush=True)
