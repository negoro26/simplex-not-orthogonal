"""Synthetic polysemy corpus generator for the sense-allocation study.

Controls:
  - sense frequencies: Zipf within each ambiguous word (rank 1..k)
  - cue-sharing structure: disjoint | blocks | shared
    (sets which senses share context-cue tokens -> cross-word co-activation)
Constraints enforced by construction:
  - one target word appears at most once per sentence
    -> same-word senses never co-activate (perfect mutual exclusivity)
  - continuation token after the main target is predictable only from
    (cue, target) -> loss requires encoding the sense
"""
import random
from collections import Counter, defaultdict


def build(grouping="blocks", n_words=12, k=4, n_blocks=3, cues_per_pool=24,
          n_fillers=80, sents_per_word=1600, zipf_pow=1.0, seed=7,
          pool_by="rank"):
    """pool_by="rank": one cue pool per (block, rank) -- cue tokens carry rank.
    pool_by="group": one cue pool per block shared by all its ranks -- cue
    tokens carry no rank information (registered 2026-09-02 for the
    rank-mixed control; with k not dividing n_blocks, blocks mix ranks)."""
    rng = random.Random(seed)
    ranks = list(range(1, k + 1))
    pw_raw = [r ** -zipf_pow for r in ranks]
    z = sum(pw_raw)
    pw = [p / z for p in pw_raw]                      # P(sense rank r)

    senses = [(w, r) for w in range(n_words) for r in ranks]
    N = len(senses)
    sid = {sn: i for i, sn in enumerate(senses)}
    if grouping == "disjoint":
        gid = list(range(N))
    elif grouping == "shared":
        gid = [0] * N
    elif grouping == "blocks":
        gid = [i % n_blocks for i in range(N)]
    else:
        raise ValueError(grouping)
    G = max(gid) + 1

    if pool_by == "group":
        shared_pool = [{f"c{g}_{j}" for j in range(cues_per_pool)} for g in range(G)]
        pools = [[shared_pool[g] for _r in ranks] for g in range(G)]
    else:
        pools = [[{f"c{g}_{r}_{j}" for j in range(cues_per_pool)} for r in ranks]
                 for g in range(G)]
    fillers = [f"f{j}" for j in range(n_fillers)]
    cont = {sn: f"cont_w{sn[0]}_{sn[1]}" for sn in senses}

    def sample_sense(exclude):
        for _ in range(1000):
            w = rng.randrange(n_words)
            if w not in exclude:
                return (w, rng.choices(ranks, weights=pw)[0])
        raise RuntimeError("could not sample distinct word")

    corpus, senses_out = [], []
    main_count = Counter()
    cooc = Counter()
    cue_seen = defaultdict(set)
    for w in range(n_words):
        for r in ranks:
            for _ in range(round(sents_per_word * pw[r - 1])):
                m = rng.randint(2, 4)
                main_slot = rng.randrange(m)
                sent, senses_in = ["<bos>"], []
                in_sentence = {w}
                for slot in range(m):
                    if slot == main_slot:
                        tw, tr = w, r
                    else:
                        tw, tr = sample_sense(in_sentence)
                    in_sentence.add(tw)
                    g = gid[sid[(tw, tr)]]
                    cue = rng.choice(sorted(pools[g][tr - 1]))
                    sent.append(cue)
                    sent.append(f"w{tw}")
                    if slot == main_slot:
                        sent.append(cont[(tw, tr)])
                        main_count[(tw, tr)] += 1
                        cue_seen[(tw, tr)].add(cue)
                    sent.append(rng.choice(fillers))
                    senses_in.append((tw, tr))
                corpus.append(sent)
                for a in range(len(senses_in)):
                    for b in range(a + 1, len(senses_in)):
                        cooc[tuple(sorted((senses_in[a], senses_in[b])))] += 1
                senses_out.append(senses_in)
    return dict(corpus=corpus, senses_of=senses_out, main_count=main_count,
                cooc=cooc, cue_seen=cue_seen, pw=pw, ranks=ranks)


def jaccard_stats(cue_seen, senses, gid):
    """Returns (same-rank-within-group, cross-group) mean Jaccards.
    After rank-slicing, cue overlap is expected ONLY for pairs sharing
    both group and rank (same slice used by different words)."""
    same_rank_w, cross = [], []
    for i in range(len(senses)):
        for j in range(i + 1, len(senses)):
            a, b = cue_seen[senses[i]], cue_seen[senses[j]]
            if not a or not b:
                continue
            jac = len(a & b) / len(a | b)
            if gid[i] != gid[j]:
                cross.append(jac)
            elif senses[i][1] == senses[j][1]:
                same_rank_w.append(jac)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return mean(same_rank_w), mean(cross)


def check(grouping, **kw):
    d = build(grouping=grouping, **kw)
    total = sum(d["main_count"].values())
    # 1. Zipf recovery (pooled across words per rank)
    worst = max(abs(sum(d["main_count"][(w, r)] for w in range(kw["n_words"]))
                    / total - d["pw"][r - 1]) for r in d["ranks"])
    # 2. same-word senses never co-occur
    same_word = sum(v for (a, b), v in d["cooc"].items() if a[0] == b[0])
    # 3. cue-sharing structure
    senses = sorted(d["cue_seen"])
    gid = None
    N = len(senses)
    if grouping == "disjoint":
        gid = list(range(N))
    elif grouping == "shared":
        gid = [0] * N
    else:
        gid = [i % kw.get("n_blocks", 3) for i in range(N)]
    wi, cr = jaccard_stats(d["cue_seen"], senses, gid)
    ok = worst < 0.03 and same_word == 0
    if grouping == "blocks":
        ok &= cr == 0.0 and wi > 0.9
    elif grouping == "shared":
        ok &= wi > 0.9
    else:  # disjoint
        ok &= cr == 0.0
    print(f"{grouping:>8}: freq_err={worst:.4f}  same_word_cooc={same_word}  "
          f"jaccard_within={wi:.3f}  across={cr:.3f}  "
          f"{'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    for g in ("disjoint", "blocks", "shared"):
        check(g, n_words=12, k=4, n_blocks=3)
