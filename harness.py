"""Corpus -> training -> activation capture harness."""
import functools
import random
import time

import torch
import torch.nn.functional as F

from gen import build
from model import PAD, TinyLM


def parse_cont(tok):
    """'cont_w3_2' -> (3, 2)"""
    _, w, r = tok.split("_")
    return int(w[1:]), int(r)


@functools.cache
def prepare(seed=7, sents_per_word=1600, **gkw):
    d = build(sents_per_word=sents_per_word, seed=seed, **gkw)
    corpus, senses_of = d["corpus"], d["senses_of"]
    rng = random.Random(seed + 999)
    order = list(range(len(corpus)))
    rng.shuffle(order)
    test_set = set(order[: len(order) // 10])

    toks = sorted({t for s in corpus for t in s})
    stoi = {t: i + 1 for i, t in enumerate(toks)}
    cont_id = {stoi[t]: parse_cont(t) for t in stoi if t.startswith("cont_")}

    def enc(i):
        s = corpus[i]
        ids = [stoi[t] for t in s]
        tgt = ids[1:] + [PAD]
        mask = [float(s[j + 1].startswith("cont_")) for j in range(len(ids) - 1)] + [0.0]
        return ids, tgt, mask

    train = [enc(i) for i in range(len(corpus)) if i not in test_set]
    test = [enc(i) for i in sorted(test_set)]
    meta = dict(V=len(stoi) + 1, cont_id=cont_id,
                test_senses=[senses_of[i] for i in sorted(test_set)])
    return train, test, meta, d


def pad_rows(rows):
    """(ids, tgt, mask) rows -> x, y, m padded to the longest row."""
    T = max(len(r[0]) for r in rows)
    x = torch.full((len(rows), T), PAD, dtype=torch.long)
    y = torch.zeros(len(rows), T)
    m = torch.zeros(len(rows), T)
    for b, (ids, tgt, mask) in enumerate(rows):
        x[b, :len(ids)] = torch.tensor(ids)
        y[b, :len(tgt)] = torch.tensor(tgt)
        m[b, :len(mask)] = torch.tensor(mask)
    return x, y, m


def make_batch(rows, bs, rng):
    return pad_rows([rows[rng.randrange(len(rows))] for _ in range(bs)])


def train(V=None, data=None, d_model=64, gamma=0.0, steps=1500, bs=256,
          lr=1e-3, seed=0, log_every=500, lam=0.0, word_of=None,
          device="cpu", prec="fp32", pen="coact", freq_of=None):
    """lam > 0 adds the intervention penalty: mean squared cosine between
    resid vectors (at prediction positions) whose targets are continuation
    tokens of DIFFERENT words. Same-word pairs excluded — never fights the
    antipodal optimum.
    pen="coact": pairs enter each batch at their empirical co-occurrence
    rate (the registered lambda arm).
    pen="uniform": inverse-frequency weights 1/(f_si*f_sj) make every
    cross-word pair type's EXPECTED per-step price equal — prices shared-
    cue / rare-rank pairs directly instead of at their Zipf co-activation
    rate. freq_of: FloatTensor[V], corpus sense frequency by token id."""
    torch.manual_seed(seed)
    fast = str(device).startswith("cuda")
    model = TinyLM(V, d_model).to(device)
    if word_of is not None:
        word_of = word_of.to(device)
    if freq_of is not None:
        freq_of = freq_of.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=gamma,
                            fused=fast)
    rng = random.Random(seed)
    t0 = time.time()
    for step in range(steps):
        x, y, m = (t.to(device) for t in make_batch(data, bs, rng))
        if fast:
            # bf16 autocast (Amendment 5, gated): block + head in bf16,
            # fp32 master weights; penalty statistics below are forced
            # back to fp32. Default stays the registered TF32 path.
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(fast and prec == "bf16")):
                resid = model.features(x)
                sel = m[:, :-1] > 0
                hs = resid[:, :-1][sel]
                ys = y[:, :-1][sel].long()
                loss = F.cross_entropy(model.unemb(hs), ys,
                                       reduction="none").sum() \
                    / m[:, :-1].sum()
        else:
            logits, streams = model(x)
            losses = F.cross_entropy(logits[:, :-1].reshape(-1, V),
                                     y[:, :-1].reshape(-1).long(),
                                     reduction="none").reshape(y[:, :-1].shape)
            loss = (losses * m[:, :-1]).sum() / m[:, :-1].sum()
            resid = streams["resid"]
        if lam:
            # penalty statistics always fp32: cos^2 in reduced precision
            # is not acceptable for a term that shapes geometry (Amendment
            # 5 gate 3); on the fp32 path the wrapper and cast are no-ops.
            with torch.autocast("cuda", enabled=False):
                w = word_of[y[:, :-1].long()]
                sel = (m[:, :-1] > 0) & (w >= 0)
                H = F.normalize(resid[:, :-1][sel].float(), dim=-1)
                ww = w[sel]
                cross = ww[:, None] != ww[None, :]
                c2 = (H @ H.T)[cross] ** 2
                if pen == "uniform":
                    fi = freq_of[y[:, :-1][sel].long()].float()
                    wij = 1.0 / (fi[:, None] * fi[None, :])
                    loss = loss + lam * (c2 * wij[cross]).sum() \
                        / wij[cross].sum()
                else:
                    loss = loss + lam * c2.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:5d}  loss {loss.item():.4f}", flush=True)
    return model, time.time() - t0


@torch.no_grad()
def capture(model, data, meta, bs=512):
    """Sense vectors at the three candidate readout sites, held-out corpus.

    Also returns by_rank: rank -> dict(acc, loss, n) over masked positions —
    the behavioral endpoint (rare-sense accuracy vs frequent-sense loss)."""
    rng = random.Random(0)
    caps = {s: {} for s in ("emb", "attn_raw", "resid")}
    acc_n = acc_ok = 0
    br = {}
    for off in range(0, len(data), bs):
        rows = data[off:off + bs]
        x, y, m = make_batch(rows, len(rows), rng)
        logits, streams = model(x)
        V = logits.shape[-1]
        hit = (logits[:, :-1].argmax(-1) == y[:, :-1]) * m[:, :-1]
        acc_ok += hit.sum().item()
        acc_n += m[:, :-1].sum().item()
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, V),
                             y[:, :-1].reshape(-1).long(),
                             reduction="none").reshape(y[:, :-1].shape)
        for b in range(x.shape[0]):
            for t in range(x.shape[1] - 1):
                nxt = int(y[b, t].item())
                if m[b, t].item() and nxt in meta["cont_id"]:
                    w, r = meta["cont_id"][nxt]
                    for name in caps:
                        caps[name].setdefault(w, {}).setdefault(r, []).append(
                            streams[name][b, t])
                    s = br.setdefault(r, [0, 0.0, 0])
                    s[0] += int(hit[b, t].item())
                    s[1] += ce[b, t].item()
                    s[2] += 1
    by_rank = {r: dict(acc=ok / n, loss=ls / n, n=n)
               for r, (ok, ls, n) in br.items()}
    return caps, acc_ok / acc_n, by_rank
