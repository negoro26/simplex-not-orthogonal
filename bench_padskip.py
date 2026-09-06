"""Gate for model.py pad skipping: fingerprint vs the dense path, then
throughput. Full arm-5 shape (d=4096, bs=256), fixed batch, seed 0.
Usage: python bench_padskip.py [steps=40]"""
import random
import sys
import time

import torch
import torch.nn.functional as F

import harness
from model import TinyLM

torch.backends.cuda.matmul.allow_tf32 = True
steps = int(dict(kv.split("=") for kv in sys.argv[1:]).get("steps", 40))
BIG = dict(n_words=2000, k=5, n_blocks=500, cues_per_pool=4,
           n_fillers=80, sents_per_word=48)
train_data, _, meta, _ = harness.prepare(seed=0, **BIG)
V = meta["V"]
dev = "cuda"
torch.manual_seed(0)
model = TinyLM(V, 4096).to(dev)
rng = random.Random(0)
x, y, m = (t.to(dev) for t in harness.make_batch(train_data, 256, rng))
keep = x != 0
print(f"batch tokens {x.numel()}  real {int(keep.sum())}  "
      f"({100 * (1 - keep.sum().item() / x.numel()):.1f}% pad)")


def step_loss(resid):
    sel = m[:, :-1] > 0
    hs = resid[:, :-1][sel]
    ys = y[:, :-1][sel].long()
    return F.cross_entropy(model.unemb(hs), ys, reduction="none").sum() \
        / m[:, :-1].sum()


def fingerprint(dense, bf16):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        e = model.emb(x)
        resid = model.block(e, None)[0] if dense else model.features(x)
        loss = step_loss(resid)
    loss.backward()
    grads = {n: p.grad.norm().item() for n, p in model.named_parameters()}
    return loss.item(), grads, resid.detach().float()


for bf16 in (False, True):
    l0, g0, r0 = fingerprint(True, bf16)
    l1, g1, r1 = fingerprint(False, bf16)
    gd = max(abs(g0[n] - g1[n]) / max(g0[n], 1e-30) for n in g0)
    rd = ((r0 - r1)[keep].norm() / r0[keep].norm()).item()
    print(f"[{'bf16' if bf16 else 'fp32/tf32'}] loss dense {l0:.6f} skip {l1:.6f} "
          f"rel {abs(l0 - l1) / l0:.2e}  max grad-norm rel {gd:.2e}  "
          f"real-token resid rel {rd:.2e}  pad resid max |.| {r1[~keep].abs().max():.1e}")

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
for name, dense in (("dense", True), ("skip", False)):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    for i in range(steps + 5):
        if i == 5:
            torch.cuda.synchronize()
            t0 = time.time()
        xb, yb, mb = (t.to(dev) for t in harness.make_batch(train_data, 256, rng))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            resid = model.block(model.emb(xb), None)[0] if dense else model.features(xb)
            sel = mb[:, :-1] > 0
            loss = F.cross_entropy(model.unemb(resid[:, :-1][sel]),
                                   yb[:, :-1][sel].long(), reduction="none").sum() \
                / mb[:, :-1].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    print(f"{name:>5}: {dt:.3f} s/step  -> {dt * 6000 / 60:.1f} min / 6000 steps  "
          f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f}GB alloc "
          f"{torch.cuda.max_memory_reserved() / 1e9:.2f}GB reserved", flush=True)
