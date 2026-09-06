"""Does gen.py produce sentences that actually pose the disambiguation task?

Checks at the arm-5 corpus (and toy) that the self-checks in gen.py do not:
  1. (cue, word) -> continuation is a FUNCTION (deterministic, unique)
  2. word alone is ambiguous (k continuations); cue alone is ambiguous (many)
  3. prior-only ceiling: always guessing the rank-1 sense
  4. order-free solvability: with no positional code the model sees the
     prefix as a bag; the main cue is identifiable only if no distractor cue
     belongs to a group that also contains the target word
  5. test rows whose (word, rank) sense has ZERO train examples (unlearnable
     unembedding row) and exact train/test duplicate sentences
  6. a few raw sentences, so a human can read them
Usage: python check_gen.py [big|toy] [blocks=N] [pool=rank|group]
"""
import random
import sys
from collections import Counter, defaultdict

from gen import build

BIG = dict(n_words=2000, k=5, n_blocks=500, cues_per_pool=4,
           n_fillers=80, sents_per_word=48)
TOY = dict(n_words=12, k=4, n_blocks=3, cues_per_pool=24,
           n_fillers=80, sents_per_word=1600)
args = sys.argv[1:] or ["big"]
cfg = dict(TOY if args[0] == "toy" else BIG)
for kv in args[1:]:                       # e.g. blocks=499 pool=group
    k_, v_ = kv.split("=")
    cfg[{"blocks": "n_blocks", "pool": "pool_by"}[k_]] = int(v_) if v_.isdigit() else v_
seed = 0
d = build(seed=seed, **cfg)
corpus, senses_of = d["corpus"], d["senses_of"]
k = cfg["k"]

# same split rule as harness.prepare
rng = random.Random(seed + 999)
order = list(range(len(corpus)))
rng.shuffle(order)
test = set(order[: len(order) // 10])

print(f"corpus: {len(corpus)} sentences, {len(test)} test; "
      f"lengths {sorted(Counter(len(s) for s in corpus).items())}")
print("samples:")
for i in (0, 1, len(corpus) // 2, len(corpus) - 1):
    print("  ", " ".join(corpus[i]))

def main_chunk(s):
    """(cue, word, cont, position of word, prefix cues) for the main slot."""
    j = next(j for j, t in enumerate(s) if t.startswith("cont_"))
    cues = [t for t in s[:j - 2] if t.startswith("c")]
    return s[j - 2], s[j - 1], s[j], j - 1, cues


pair_to_cont, word_to_cont, cue_to_cont = defaultdict(set), defaultdict(set), defaultdict(set)
main_rank, train_count = Counter(), Counter()
for i, s in enumerate(corpus):
    cue, w, cont, _, _ = main_chunk(s)
    pair_to_cont[(cue, w)].add(cont)
    word_to_cont[w].add(cont)
    cue_to_cont[cue].add(cont)
    r = int(cont.split("_")[-1])
    main_rank[r] += 1
    if i not in test:
        train_count[(w, r)] += 1

# 1-2: determinism and ambiguity
assert all(len(v) == 1 for v in pair_to_cont.values()), "(cue, word) not a function"
print(f"1. (cue, word) -> cont deterministic for all {len(pair_to_cont)} pairs")
wa = Counter(len(v) for v in word_to_cont.values())
ca = Counter(len(v) for v in cue_to_cont.values())
print(f"2. continuations per word: {dict(wa)}  (k={k} expected);  "
      f"per cue: min {min(ca)} max {max(ca)}  (= senses sharing the cue's pool)")
assert all(n == k for n in wa), "some word has != k continuations"

# 3: prior-only ceiling
tot = sum(main_rank.values())
print(f"3. rank frequencies {[round(main_rank[r] / tot, 3) for r in range(1, k + 1)]}; "
      f"prior-only (always rank 1) ceiling {main_rank[1] / tot:.3f}; chance 1/{cfg['n_words'] * k}")

# 4: bag ambiguity — a distractor cue that forms a valid (cue, word) pair
# with the target word, yielding a DIFFERENT continuation. With no positional
# code and one causal layer the prefix is a bag, so these are unresolvable;
# the best the model can do is pick the most frequent candidate rank.
conflict, per_rank, ceil_hit = 0, Counter(), Counter()
for s in corpus:
    cue, w, cont, _, cues = main_chunk(s)
    r = int(cont.split("_")[-1])
    alts = {next(iter(pair_to_cont[(c, w)])) for c in cues if (c, w) in pair_to_cont}
    alts.discard(cont)
    if alts:
        conflict += 1
        per_rank[r] += 1
        ranks = [r] + [int(a.split("_")[-1]) for a in alts]
        ceil_hit[r] += int(min(ranks) == r)     # best guess = most frequent rank
    else:
        ceil_hit[r] += 1
print(f"4. order-free CONFLICTS: {conflict}/{len(corpus)} sentences "
      f"({100 * conflict / len(corpus):.2f}%) have a distractor cue that pairs "
      f"with the target word into a different continuation; by rank "
      f"{dict(sorted(per_rank.items()))}")
print(f"   best-guess accuracy ceiling for an order-blind reader: overall "
      f"{sum(ceil_hit.values()) / len(corpus):.3f}; by rank "
      f"{ {r: round(ceil_hit[r] / main_rank[r], 3) for r in range(1, k + 1)} }")

# 5: unlearnable test rows and duplicates
zero = Counter()
for i in sorted(test):
    _, w, cont, _, _ = main_chunk(corpus[i])
    r = int(cont.split("_")[-1])
    if train_count[(w, r)] == 0:
        zero[r] += 1
test_by_rank = Counter(int(main_chunk(corpus[i])[2].split("_")[-1]) for i in test)
print(f"5. test rows per rank {dict(sorted(test_by_rank.items()))}; "
      f"rows whose sense has 0 train examples {dict(sorted(zero.items()))} "
      f"-> r{k} reachable ceiling {(test_by_rank[k] - zero[k]) / test_by_rank[k]:.3f}")
sents = Counter(tuple(s) for s in corpus)
dups = sum(v - 1 for v in sents.values())
tr = {tuple(corpus[i]) for i in range(len(corpus)) if i not in test}
leak = sum(tuple(corpus[i]) in tr for i in test)
print(f"   exact duplicate sentences {dups}; test sentences also in train {leak}")
print(f"   senses with < 2 train examples: "
      f"{sum(1 for (w, r), c in train_count.items() if c < 2)} of {len(train_count)}")
