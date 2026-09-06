# Discovery log — arm 5: capacity-bound packing (d=4096, S=10⁴)

Running lab notebook for the large arm. Design registered BEFORE results.
Toy-scale results (d=64, capacity slack) are in the final section of this file.

## Question

At d=4096 with S=10⁴ senses (2000 words × 5 Zipf ranks), sense directions
cannot be mutually orthogonal (S > d). What packing does SGD find, and how
far is it from what random/optimal codes achieve at the same tolerance?

## Registered design (2026-08-22, before first full run)

Corpus: `build(n_words=2000, k=5, n_blocks=500, cues_per_pool=4,
n_fillers=80, sents_per_word=48, grouping="blocks", zipf_pow=1)` →
96,000 sentences, V=14,082, max T=14. Note: k=5 divides n_blocks=500, so
blocks are rank-pure (each pool = 20 same-rank senses). Pools are
rank-indexed by construction, so sharing sets are same-rank at every scale;
no confound introduced.

Model/training: d=4096 (317M params), fp32 AdamW on GPU (RTX 5060),
bs=256, lr=1e-3 (amended below), steps=6000 (16 epochs; toy was ~40 —
mismatch noted, see caveats). γ=0 primary arm; γ/λ arms only after
baseline is understood.

Per-sense data (Zipf, per word): r1=21, r2=11, r3=7, r4=5, r5=4 sentences.
One capture vector per sentence (main slot only).

Endpoints and decision rules:
1. **Same-word geometry at capacity**: pre-registered measure.py rule on
   median cross-sense cos after per-word centering (SIMPLEX < −0.15 <
   ORTHOGONAL < +0.15 < COLLAPSED). k=5 → regular-simplex value −0.25.
2. **Cross-word packing (the headline)**: per-word-centered normalized
   sense centroids (same objects the interference measure uses). All
   cross-word pairwise |cos|; ε-grid log-spaced ±4°–10° (8 points);
   violation curve viol(ε) = fraction of pairs with |cos| > sin ε;
   empirical crossover ε* = median angular deviation from 90°. Compared
   against a matched random baseline: same count, same d, unit Gaussian
   directions, same statistics. The SGD-vs-random and SGD-vs-bound gaps
   are the result; no bound constant pre-committed.
3. **Behavior**: per-rank continuation accuracy on the held-out 10% test
   split. r5 (4 sents/word) probes compositional generalization: cue
   carries rank, target carries word, so unseen (cue, word) pairs are
   inferable in principle.
4. **Frequency axis**: per_rank_stats as at toy scale; r5 centroids rest
   on ~4 vectors — reliability column is the honest denominator.

**Amendment (registered before first full run, after 300-step probes):**
the toy lr=1e-3 DIVERGES at d=4096 (loss 9.55 → 369 by step 200; the
reliability floor correctly collapsed to 0.07–0.40, flagging the garbage
geometry). lr=1e-4 descends smoothly (9.55 → 2.56 in 300 steps); 3e-5 is
needlessly slow. Registered: lr=1e-4 for all arm-5 runs. No grad clipping,
no warmup — no new knobs.

**Amendment 2 (registered 2026-08-23, before any full run completes):**
TF32 enabled for arm-5 training and capture matmuls (Blackwell tensor
cores; ~3-4× wall-time on the dominant fp32 GEMMs). Rounding is ~1e-3
relative — far below the centroid noise floor that split-half reliability
measures (0.83-0.98 at toy scale). Statistics-critical gram matrices in
`packing_report` run on CPU in strict fp32, unaffected. Toy-scale runs are
CPU and untouched; cross-scale comparisons already cross numeric backends.

**Amendment 3 (registered 2026-08-23, before any full run completes):**
8GB Blackwell fit (RTX 5060 target): CUDA arm-5 training switches to a
chunked cross-entropy head (2048-position chunks, each recomputed via
activation checkpointing alongside the block itself) and fused AdamW. The
chunked loss is per-chunk masked sums over the global mask count — the same
masked mean up to fp32 addition order (~1e-7 relative, well inside the
Amendment 2 envelope). Checkpointing recomputes the block bit-for-bit
(no dropout, no RNG anywhere in Block). Fused AdamW is the identical
elementwise update in one kernel. Full-batch logits never materialize
during training; capture (no-grad) is unaffected; CPU toy-scale runs keep
the untouched eager path.

**Amendment 4 (registered 2026-08-24, before the seed-3 replication run):**
wall-time fix, no new knobs, every claim GPU-measured at the full arm-5
shape on an idle RTX 5060. Exactly one continuation token exists per
sentence by construction (mask density 256/3328 = 7.69% at bs=256), so the
CUDA head now gathers the masked rows before the unembedding instead of
unembedding all B·(T−1) positions and multiplying by the mask. The dropped
terms are exact zeros (the mask is float(bool)) and the divisor is
unchanged: step-0 loss is BITWISE identical before/after and the largest
one-step per-parameter grad-norm shift is 2.0e-5 relative — 50× inside the
Amendment 2 envelope. With the head transient down from ~350MB to 14MB,
block activation checkpointing no longer buys peak memory and is removed;
measured bitwise exact on the gradient fingerprint (delta exactly 0.0 — it
only deletes a recompute of an RNG-free block). Peak VRAM moves 5.79 →
5.97GB allocated (6.00 → 6.34GB reserved) and throughput 0.436 → 0.291
s/step; VRAM preflight updated to the measured peaks. Capture unembeds the
same selected positions under no_grad; verified end-to-end: reanalyzing
`results/big_g0.0_l0.0_s0.ckpt` reproduces the stored artifact BITWISE
(all 10000 centroids, packing dict, med, interference, acc, test_rank),
and every printed log line matches the recorded seed-0 run. Batch
construction is unchanged (its cost measured fully hidden; only the
duplicated padding code was unified). Runs before and after this amendment
are comparable on the same footing as Amendments 2–3.

**Amendment 5 (registered 2026-08-24, both gates measured before any bf16
science run):** bf16 autocast adopted for arm-5 CUDA training. GATE 1
(one-step fingerprints at full shape, fixed batches): step-0 loss rel
delta 2.0e-7 (λ=0) / 5.5e-6 (λ=1); max per-parameter grad-norm delta
1.5e-3 / 1.8e-3 — the predicted bf16 envelope, NOT inside TF32's, hence
this separate registration. Throughput 0.289 → 0.172 s/step (1.68×;
~17 min train), peak memory unchanged (~6.0GB allocated / 6.34GB
reserved). The λ penalty statistics are FORCED TO FP32 under autocast —
cos² in reduced precision is not acceptable for a term that shapes
geometry. GATE 2 (full bf16-trained seed 0; capture/analysis stay strict
fp32): split-half reliability {r1..r5} = 0.885/0.790/0.675/0.557/0.486
vs the TF32 run's 0.884/0.784/0.673/0.555/0.480 — within 0.006 on every
rank; held-out acc 0.908, same-word median cos −0.250 SIMPLEX,
interference 0.0235, SGD p50 dev 5.44° vs random 0.60° — all inside the
seed spread. Both gates PASS: bf16 is the default fast path for
subsequent arm-5 runs; artifacts carry a `_bf16` suffix and record
`prec` in their config.

**Amendment 6 (registered 2026-09-02, gate measured before any further
arm-5 run):** pad skipping in `model.py` (CUDA only). Corpus rows are
8/11/14 tokens, batches pad to 14, so 22.4% of block positions are pad;
attention is causal with no positional code and every reader gates on the
mask, so pad outputs are dead. Per-token GEMMs (qkv, out, MLP) now run on
real tokens only; the T×T attention keeps the padded layout with pad
q/k/v = 0; pad outputs are returned as exact zeros. GATE (`bench_padskip.py`,
full shape, fixed batch, seed 0): loss BITWISE identical under both TF32
and bf16 autocast; real-token residuals bitwise identical; max
per-parameter grad-norm rel delta 7.8e-7 (TF32) / 9.7e-8 (bf16) — fp32
summation order only, 2000× inside the Amendment 5 envelope. Throughput
0.173 → 0.153 s/step (~17 → ~15 min train); peak VRAM 6.18GB allocated /
6.61GB reserved (unchanged). Capture and `ablate.py` inherit the same path.
CUDA-event split of the remaining step: block fwd 26%, bwd 52%, fused
AdamW 18% (memory-bound: 317M params × 7 fp32 words ≈ 8.9GB/step at the
GDDR7 bus ceiling), batch build 2%. GEMMs sustain ~27 TFLOPS bf16 on this
power-limited laptop part; no further software lever exists that does not
alter numerics (bf16 optimizer states, step count, batch size — all
protocol knobs, refused).

**Run robustness (no statistics impact):** weights checkpoint to
`results/big_*.ckpt` immediately after training; `bigrun.py ckpt=<path>`
re-runs analysis only (verified round-trip). VRAM preflight asserts free
memory before training (6.5GB train / 2.5GB analysis) so an occupied GPU
fails in seconds, not mid-hour. MEASURED (2026-08-23, 60-step benchmark at
full arm-5 shape, TF32 + chunked CE + checkpointing + fused AdamW):
0.435 s/step at bs=256 → 43 min train; peak VRAM 5.90GB allocated (1.6GB
headroom on the 8GB card). Chunked path verified against eager: loss rel
delta 7.7e-8, max grad rel delta 2.8e-7 — inside the Amendment 3 claim.
Post-Amendment-4 re-benchmark (2026-08-24, integrated `harness.train`, 60
steps at full shape): 0.291 s/step at bs=256 → ~29 min train; peak 6.03GB
allocated / 6.34GB reserved.

Capture protocol deviation (registered): geometry is measured on the FULL
corpus (train+test), not the 10% test split — at 48 sents/word the test
split leaves <1 vector for rare senses. The toy-arm caveats (end of file) already bless
seen-data geometry for cross-arm comparisons. Behavioral accuracy stays
held-out. Interference weights come from full-corpus co-occurrence.
Only the resid stream is captured (primary site; emb is vacuous, attn_raw
secondary) — 96k × 4096 × 3 streams would be 4.7GB for no added claim.

## Status 2026-08-23 (superseded): seed 0 attempts before the perf fix

- Attempt 1 (plain shell): healthy. lr=1e-4, loss 9.55 → ~0.005–0.02 by
  step 5500 (train task solved). Killed at 60 min by the tool's 3600s
  timeout, minutes before completion. Wall ~0.6 s/step at bs=256 (thermal
  throttle; probe said 0.45).
- Attempts 2 and 3 (hub-supervised): exit 137 — the USER stopped them
  deliberately. No supervisor/cgroup problem; earlier suspicion withdrawn.
- Paused here at user request. To resume: `bigrun.py seed=0 steps=6000` —
  measured 43 min train + ~3 min capture/analysis, fits one <50 min window;
  if anything interrupts after training,
  `bigrun.py seed=0 ckpt=results/big_g0.0_l0.0_s0.ckpt` redoes analysis
  only. Attempt 1 showed the run itself is healthy.

Artifacts: per-run centroids saved to `results/big_*.pt` for reanalysis;
logs to `results/big_*.log`.

## Results

### Seed 0 (γ=0, λ=0, 6000 steps) — completed 2026-08-23, 45 min wall

Log: `results/big_g0.0_l0.0_s0.log`; centroids: `results/big_g0.0_l0.0_s0.pt`.
Train loss 9.55 → 0.002; held-out cont acc 0.900.

**1. Same-word geometry at capacity: SIMPLEX, and exact.** Median
cross-sense cos −0.249 against the k=5 regular-simplex value −0.25
(n=20,000 pairs). Rank-pair matrix near-uniform (−0.23…−0.30): the
toy-scale rare-rank merge is ABSENT at capacity — merging was a d=64
crowding effect, senses of a word have room at d=4096. Split-half
reliability: r1 0.88, r2 0.78, r3 0.67, r4 0.56, r5 0.48 (r5 rests on
4 sents/word; the honest denominator registered in the design).

**2. Cross-word packing (headline): SGD does NOT approach random-level
orthogonality, let alone ETF.** Angular deviation from 90°: SGD median
5.60° vs matched-random 0.60° — ~9× broader at every quantile; SGD
p99.9 = 54.2°, max 73°, where random maxes at 5.2°. viol(SGD) ≥ 0.22 on
the whole ±4°–10° ε-grid where viol(random) = 0.0000.

**3. Behavior:** held-out per-rank acc 0.983 / 0.940 / 0.859 / 0.756 /
0.617 (r1→r5), losses 0.06→1.52 — the graded Zipf ladder, task solved.
Weighted cross-word interference 0.0244 (pair mass 1.00) vs 0.33 at
toy scale γ=0 — capacity buys a 13× drop.

### Post-hoc decomposition (seed 0 only, NOT pre-registered)

The deviation mass splits into two populations by a corpus covariate:

- **Co-occurrence is NOT the driver.** Pairs that co-occur in sentences
  (317,619; base 6.4e-3) show no meaningful enrichment in the tail (1.5%
  at p99.9, ×2) and near-identical mean deviation (7.16° vs 6.63°).
- **Shared cue context IS the driver.** Cross-word pairs whose senses
  were trained under ≥1 common cue token (94,328 = 0.19% of pairs):
  median deviation 55° (|cos| ≈ 0.82, near-collinear),
  P(dev>21°|shared) = 1.000, P(dev>54°|shared) = 0.53. The extreme tail
  (>54.2°, n=50,067) is 100% shared-cue and exclusively same-rank (cue
  pools live within (block, rank)), dominated by FREQUENT ranks
  (r1-r1: 19k, r2-r2: 17k of 50k) — most training signal, most alignment.
- **Context-disjoint pairs remain ~9× broader than random:** median
  5.58°, p99.9 25.8°, max 54.7° (random: 0.60°, 2.9°, 5.2°). The gap to
  random is a global property of SGD's solution, not explained by cue
  sharing.
- **Dose-response confirmed (deviation vs cue-set Jaccard):** among
  shared-cue pairs, median deviation rises monotonically with context
  overlap — J≤0.25: 43.0°, ≤0.5: 47.4°, ≤0.75: 52.6°, ≤1.0: 61.2°
  (p90: 51.9° → 66.5°). All 94,328 shared-cue pairs are same-rank; cue
  reuse never crosses ranks in this corpus. Alignment is graded in
  shared-context fraction, not a threshold effect.

### Seeds 1–3 replication — completed 2026-08-23/24

Every endpoint replicates within noise across all four seeds (seed 3 is
the first run on the Amendment-4 fast path, 30 min wall):

| endpoint | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| same-word median cos (simplex −0.25) | −0.249 | −0.250 | −0.250 | −0.250 |
| held-out cont acc | 0.900 | 0.910 | 0.901 | 0.905 |
| weighted interference | 0.0244 | 0.0234 | 0.0257 | 0.0234 |
| SGD dev p50 / p99.9 (random 0.60/2.9°) | 5.60° / 54.2° | 5.43° / 54.8° | 5.65° / 54.5° | 5.41° / 54.7° |
| dev median, shared-cue \| disjoint | 55.1° \| 5.58° | 55.6° \| 5.41° | 55.3° \| 5.64° | 55.6° \| 5.40° |
| P(dev>21° \| shared / disjoint) | 1.000 / 0.0082 | 1.000 / 0.0067 | 1.000 / 0.0087 | 0.999 / 0.0064 |
| Jaccard dose-response medians | 43/47/53/61° | 43/48/53/62° | 43/48/53/62° | 43/48/53/62° |

Rank-pair matrix uniform in all four (no merge); reliability ladders
identical. Post-hoc decomposition codified in `posthoc.py` (pure CPU
reanalysis of saved centroids).

Reading: the model spends representation alignment on *feature sharing*
(a common cue context becomes a common direction component across words)
— exactly on pairs that rarely co-occur, so co-occurrence-weighted
interference stays low and behavior is unharmed. Orthogonal packing is
not what SGD buys with capacity; simplex-exact same-word structure plus
shared-context alignment is.

### λ arm at capacity (λ=1, seed 0) — completed 2026-08-23

The registered question: does the co-activation-weighted overlap penalty
flatten the shared-cue population, and at what behavioral price?

| endpoint | λ=0 (seed 0) | λ=1 (seed 0) |
|---|---|---|
| held-out cont acc | 0.900 | 0.904 |
| weighted interference | 0.0244 | 0.0147 (−40%) |
| dev median \| disjoint | 5.58° | 4.23° (−24%) |
| P(dev>21° \| disjoint) | 0.0082 | 0.00081 (÷10) |
| dev median \| shared-cue | 55.1° | 53.5° (−3%) |
| P(dev>21° \| shared-cue) | 1.000 | 1.000 |
| same-word median cos | −0.249 | −0.250 |

**The penalty cuts interference 40% at zero behavioral cost (as at toy
scale) and collapses the disjoint-pair tail 10×, but the shared-cue
near-collinear population is essentially untouched** (median −1.5°,
P(dev>21°)=1.000, dose-response same shape shifted ~2°). Mechanistically
consistent, not mysterious: the penalty prices pairs at their empirical
batch co-activation rate, and shared-cue pairs rarely co-occur — so
feature sharing is UNPRICED, not proven load-bearing. Same-word simplex
untouched; rare rank-pairs tilt slightly more negative (r4-r5 −0.278),
echoing the toy-scale anti-merge effect.

Open next: distinguish "unpriced" from "load-bearing" — a uniformly
weighted cross-word penalty (same-word still excluded) would price
shared-cue pairs directly; if alignment survives that, it is
load-bearing for the task, not merely cheap. Also λ=3 dose at capacity.

### Uniform pricing fork resolved — alignment is LOAD-BEARING
### (λ=1, seed 0, 2026-08-24)

The "open next" intervention above, run on the Amendment-5 fast path
(bf16; gate-2-validated endpoint-neutral): `pen=uniform` replaces batch
co-activation pricing with inverse-frequency weights 1/(f_si·f_sj),
making every cross-word pair type's expected per-step price EQUAL —
rare-rank shared-cue pairs are priced directly instead of at their Zipf
rate. Same-word still excluded; λ=1 dose unchanged.

| endpoint | coact λ=1 | uniform λ=1 |
|---|---|---|
| held-out cont acc | 0.915 | 0.908 |
| weighted interference | 0.0158 | 0.0150 |
| dev median \| shared-cue | 53.74° | 53.30° |
| P(dev>21° \| shared-cue) | 1.000 | 0.999 |
| dev median \| disjoint | 4.38° | 4.20° |
| P(dev>21° \| disjoint) | 0.00110 | 0.00089 |
| same-word median cos | −0.249 | −0.248 |

**Verdict: the shared-cue near-collinear population survives direct,
uniform pricing essentially unchanged** (median −0.4°, tail probability
flat) — by the registered criterion this is LOAD-BEARING for the task,
not an artifact of cheap co-activation pricing. Uniform weighting buys
only a little extra disjoint-tail tightening (median −0.18°, tail ÷1.2).
Behavior stays at capacity-level accuracy in both arms. Single seed;
`results/big_g0.0_l1.0_s0_bf16{,_uniform}.pt` + logs recorded.

### Readout ablation: the shared-cue subspace computes rare-rank
### disambiguation (`ablate.py`, seed 0, 2026-08-24)

Causal follow-up to the two results above. From TRAIN data only: every
cross-word pair sharing a cue pool contributes its normalized pair mean
(their common component); PCA of those directions gives the shared-cue
subspace. Held-out behavior is re-evaluated with the projection onto its
top-k directions subtracted from every residual before the unembedding.
Controls at matched k: seeded random orthonormal subspace; generic top-k
PCA of all train residuals. Pre-registered predictions P1-P3 are stated
in `ablate.py`'s header, written before first execution.

| variant | k | acc | r1 | r2 | r3 | r4 | r5 | r5 loss | med cos |
|---|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.900 | 0.983 | 0.940 | 0.859 | 0.756 | 0.617 | 1.52 | −0.248 |
| shared | 1 | 0.897 | 0.991 | 0.945 | 0.840 | 0.728 | 0.586 | 1.66 | |
| shared | 2 | 0.891 | 0.987 | 0.961 | 0.872 | 0.691 | 0.483 | 2.03 | |
| shared | 8 | 0.871 | 0.987 | 0.954 | 0.826 | 0.631 | 0.423 | 2.18 | |
| shared | 16 | 0.850 | 0.990 | 0.939 | 0.787 | 0.563 | 0.353 | 2.42 | −0.253 |
| shared | 32 | 0.790 | 0.985 | 0.890 | 0.663 | 0.404 | 0.215 | 2.90 | |
| generic | 16 | 0.854 | 0.981 | 0.945 | 0.794 | 0.596 | 0.380 | 2.30 | |
| random | 32 | 0.900 | 0.984 | 0.941 | 0.857 | 0.757 | 0.615 | 1.53 | |

**P1/P3 confirmed, P2 split.** Damage is monotone in k and lands on the
ranks that need cue-carried rank information: r5 acc 0.62 -> 0.22 with
r5 loss nearly doubling by k=32, while r1/r2 are unharmed through k=16 —
r1 actually IMPROVES under ablation (0.983 -> 0.991). The simplex
survives (median cos -0.248 -> -0.253 at k=16): the damage is behavioral,
through lost context information, not destroyed same-word geometry.
Random 32-dim ablation changes nothing (0.900 throughout): the effect is
directional content, not capacity. Generic top-PCA also hurts rare ranks,
but less selectively — at k=2 only the SHARED ablation simultaneously
helps common ranks (+0.00/+0.02/+0.01) and hammers rare ones
(-0.07/-0.13); the top variance directions of the whole space evidently
overlap the shared-cue subspace substantially.

Mechanistic claim this supports: the aligned cross-word population is a
dedicated contextual-disambiguation channel — removing it helps common-
meaning readout (less noise) and devastates compositional rare-sense
prediction. Together with the uniform-pricing result above, the picture
is consistent: training-time tidiness penalties cannot flatten this
population because the task consumes it.

**Replicated 2026-08-24 across γ=0 seeds 1–2 (fp32) and both λ=1 bf16
checkpoints (coact, uniform):**

| run | base acc | r5 @k=16 | r4 @k=16 | r5 @k=32 | medcos Δ | random@32 Δacc |
|---|---|---|---|---|---|---|
| γ0 s0 (fp32) | 0.900 | 0.353 | 0.563 | 0.215 | −0.005 | ±0.000 |
| γ0 s1 (fp32) | 0.910 | 0.389 | 0.563 | 0.257 | +0.000 | −0.001 |
| γ0 s2 (fp32) | 0.901 | 0.373 | 0.568 | 0.262 | +0.003 | ±0.000 |
| λ1 bf16 coact | 0.915 | 0.367 | 0.550 | 0.232 | ±0.000 | −0.001 |
| λ1 bf16 uniform | 0.908 | 0.405 | 0.614 | 0.314 | −0.001 | +0.001 |

Every replicate shows the same signature: monotone rare-rank destruction
under shared-subspace ablation, common ranks unharmed through k=16 (r1
improves in all five runs), simplex intact, random subspaces inert.
Two secondary observations: (a) the penalized checkpoints concentrate
MORE load in this subspace — one ablated direction already costs
coact-λ1 −0.10 r5 accuracy vs −0.03 on the unpunished run, consistent
with the penalty consolidating contextual signal into fewer directions;
(b) the uniformly priced checkpoint degrades most gracefully (highest
surviving r5/r4 at every k) — equalizing prices appears to spread the
contextual signal more evenly. Logs: `results/abl_*.log`.

### Check #1 — deletion-fairness controls: FAIL by the registered rule
### (`ablate.py`, all 5 checkpoints, 2026-09-02)

Rule and bases registered in the `ablate.py` header before execution
(rule text in the `ablate.py` header). Original base/shared/generic/random rows reproduce the
2026-08-24 tables exactly on the Amendment-6 path. New rows, r5 held-out
accuracy and median principal-angle cosine to `shared` at the same k:

| run | shared k1 / rank k1 (cos) | shared k2 / rank k2 (cos) | shared k16 / cue k16 (cos) | randvar at shared-k16 energy: k', r5 |
|---|---|---|---|---|
| γ0 s0 | 0.586 / 0.588 (1.000) | 0.483 / 0.485 (0.972) | 0.353 / 0.333 (0.994) | 1656, 0.582 |
| γ0 s1 | 0.634 / 0.643 (0.999) | 0.571 / 0.548 (0.967) | 0.389 / 0.465 (0.994) | 1623, 0.606 |
| γ0 s2 | 0.634 / 0.643 (0.999) | 0.545 / 0.534 (0.990) | 0.373 / 0.360 (0.994) | 1670, 0.589 |
| λ1 coact | 0.559 / 0.559 (0.999) | 0.498 / 0.486 (0.958) | 0.367 / 0.345 (0.982) | 1049, 0.630 |
| λ1 uniform | 0.607 / 0.608 (0.999) | 0.531 / 0.505 (0.924) | 0.405 / 0.407 (0.982) | 1018, 0.601 |

Baseline r5: 0.617 / 0.634(s1) / 0.634(s2) / 0.62 / 0.62.

1. **The top shared-cue direction IS the rank-mean direction.** In 5/5
   checkpoints the first `shared` PC and the first `rank` singular vector
   coincide (cos 0.999–1.000) and produce the same damage; the second
   directions overlap 0.92–0.99. This is a corpus fact, not a model fact:
   cue pools are rank-indexed, every shared-cue pair is same-rank, so the
   common component of shared-cue pairs is the rank code.
2. **`cue` reproduces `shared` for k ≥ 4** (cos 0.98–0.99 at k=16, damage
   within the seed spread: shared r5@16 0.353–0.405 vs cue 0.333–0.465).
   The "shared-cue subspace" is the cue-conditioned-mean subspace under a
   different estimator. `generic` top-PCA overlaps it 0.96–0.99 as well.
3. **The damage is directional, not energetic** — the one part that holds.
   Random subspaces removing the same train variance (k' ≈ 1000–1700
   dimensions for shared-k16) leave r5 at 0.58–0.63 where the 16 shared
   dimensions take it to 0.35–0.41; at k ≤ 32 random is inert.

Verdict: FAIL — `rank` reproduces the shared curve exactly where it can
(k ≤ 2) and `cue` reproduces it for k ≥ 4, with basis overlap ≥ 0.98.
"Dedicated contextual-disambiguation channel" is retracted. What the
ablation shows is that the model encodes rank/context as a low-dimensional
linear code shared across words, and deleting that code removes
rank-dependent readout — which is what any rank-informative basis does.
Per the tripwire, Check #2 (penalty-concentration retrains) is NOT run.

Reframing that survives (all measured, none retracted): same-word senses
are simplex-exact at capacity; cross-word directions sit ~9× further from
90° than a random code; the entire far tail is the shared linear rank/
context code, graded in cue overlap; co-activation and uniform pricing
cannot flatten it because rank must be read out; energy-matched random
deletion is nearly harmless. The story is "SGD builds a shared context
code instead of an orthogonal one", not "SGD hides a special channel".
Logs: `results/abl_g0.0_l0.0_s0_c1.log`, `results/abl_c1_*.log`.

### Adversarial review round (2026-09-02): post-hoc nulls and two registered controls

Three independent reviews of the draft (`paper/review/adversarial_*.md`) raised
two validity objections that turned out to be correct, checked with `nulls.py`
(post-hoc, NOT pre-registered; seed 0, `results/nulls_s0.log`):

1. **Centering identity.** `measure.analyze_word` subtracts the unweighted mean
   of the k sense centroids. For any k equal-norm equiangular vectors the
   centered pairwise cosine is exactly −1/(k−1) whatever the raw cosine. An
   UNTRAINED model (same seed) gives per-word-centered median cos −0.249.
   So "simplex-exact −0.250" is the estimator's identity, and the
   pre-registered three-way rule (simplex / orthogonal / collapsed) cannot
   return "orthogonal" for exchangeable senses — the rule was ill-posed.
   What training actually does: raw same-word cosine 0.953 → 0.527 (senses
   differentiate from the word component); per-word base norm / sense
   deviation norm 4.90 → 1.26; rank-pair matrix from graded at init
   (−0.10 r1r2 … −0.34 r4r5, a small-sample artifact of rare-sense centroids)
   to uniform (−0.23 … −0.29). Result 1 is restated as differentiation +
   uniformity, with the −0.25 dropped as evidence.
2. **Dimensionality explains the bulk.** Protocol centroids have PR 48 (init:
   383; Gaussian 2905). Isotropic vectors in d_eff=48 dims predict median
   deviation 5.60°; observed 5.56°. A spectrum-matched Gaussian null (N(0,Σ)
   of the centroids, normalized) gives p50 5.45°, p99.9 26.3°, max 43.7°.
   Observed p99.9 52.6°, max 73.1°. So "9× less orthogonal than random" is
   "the code occupies ~50 dimensions"; the isotropic-in-R^4096 null was the
   wrong null for the bulk. The TAIL survives the spectrum-matched null by
   2× at p99.9 and 30° at the max: the shared-cue alignment is real.

Two controls the reviews asked for, registered here before running:

**Width sweep** — `bigrun.py seed=0 prec=bf16 d=<512|1024|2048>`, γ=0, same
corpus, same steps. Endpoints: PR of protocol centroids, held-out acc,
cross-word p50 and p99.9 vs both nulls, raw and centered same-word cosine.
Prediction if the paper's "capacity" framing is right: PR and tail grow with
d or accuracy drops at small d. Prediction if the framing is wrong: PR ≈ 50
and the tail are d-invariant down to 512 and accuracy is flat — then the
regime is data-scarce, not dimension-scarce, and "S > d" leaves the title.

**Rank-mixed cue pools** — `gen.build(pool_by="group", n_blocks=499)`: one cue
pool per block shared by all its ranks; with 499 ∤ 5 each block holds ~20
senses of mixed ranks, so cue tokens carry no rank and shared-cue pairs are
mostly cross-rank. Verified by `check_gen.py big blocks=499 pool=group`:
deterministic (cue, word)→cont, 0.61% conflicts, no leakage. Runs:
`bigrun.py seed={0,1,2} prec=bf16 blocks=499 pool=group`, then `ablate.py`
and `posthoc.decompose` on them. Decision rule: if shared-cue pairs still sit
at |cos| ≈ 0.8 while the rank-mean subspace no longer coincides with the
shared-cue subspace (median principal-angle cosine < 0.7 at k=1), the
alignment is a shared-CONTEXT code and Check #1's FAIL was a corpus
confound; if shared-cue alignment collapses toward the disjoint bulk and
deletion damage tracks the rank basis, it is a RANK/frequency code and the
paper says "rank", not "context", throughout.

**Outcomes (runs 2026-09-02, analysis 2026-09-05; `results/TABLES.md` T9, T10;
`results/big_d*.log`, `results/big_mixed_s*.log`, `results/abl_mixed_s*.log`,
`results/nulls_mixed_s0.log`, `results/nulls_d512.log`).**

Width sweep: capacity is binding. acc 0.722 / 0.766 / 0.854 / 0.908 and r5
acc 0.450 / 0.457 / 0.600 / 0.665 for d = 512 / 1024 / 2048 / 4096; PR 13 / 19
/ 28 / 50 (grows with d, far below d); interference 0.084 → 0.024. The bulk
is dimensionality at every width: p50 11.20 / 9.13 / 7.42 / 5.44 against null B
11.02 / 8.96 / 7.29 / 5.37. The tail is width-invariant, p99.9 55.1 / 56.4 /
58.0 / 54.6, while null B's p99.9 falls 48.7 → 25.9, so the tail's excess over
the null grows with width (1.13× → 2.1×). Same-word centered cosine −0.25 at
every d (the identity). "S > d" stays; the framing "width buys accuracy and
effective dimensions, SGD still spends ~1% of d" is what the paper says.

Rank-mixed cue pools: the alignment follows the cue. Shared-cue median
deviation 50.1–50.6°, disjoint 4.76–4.77°, P(dev > 21°) 1.000 vs
0.0028–0.0030, observed p99.9 49.6–50.1 vs null B 23.2, acc 0.909–0.915 (3
seeds). Part (a) of the decision rule is met: this is a shared CONTEXT code,
and Check #1's FAIL was a corpus confound for the *geometric* claim. Part (b)
is not met, and cannot be: the top principal direction of the shared-cue pair
means coincides with the top rank-mean direction (cos 1.000) in the mixed
corpus as well. Pair means vary in rank composition, so PCA of pair means
puts the frequency axis first whatever the pooling; that coincidence is a
property of the basis construction, not of the corpus. Consequence: the
readout-deletion result stays as registered after Check #1 (directional but
basis-agnostic: cue / generic / rare coincide with shared from k ≥ 4, cos
0.96–0.99; variance-matched random damages less at equal variance, e.g.
mixed s0 39.5% var shared acc 0.877 vs randvar 0.900). Deleting k = 1 alone
is seed-inconsistent on r5 (+0.03 / −0.035 / −0.02), not claimed.

Paper wording after this round: "context" for the geometry, "top-variance
directions" for the deletion, no mechanism named.

### Second adversarial round (2026-09-05): the alignment is inherited, not built

Three fresh reviews of the rewritten draft (`paper/review/adversarial2_*.md`)
converged on one blocker: in a one-layer bag-of-tokens model the residual at
the prediction position sums the prefix embeddings, so two senses trained under
the same cue pool share a cue-embedding component *by construction*, and the
paper had never measured shared-cue alignment on the untrained model. Run
2026-09-05 (`nulls.py`, seed 0, both corpora; post-hoc, NOT pre-registered):

| | main corpus init | main trained | mixed init | mixed trained |
|---|---|---|---|---|
| PR | 383 | 48 | 396 | 62 |
| shared-cue median (°) | 17.2 | 53.5 | 15.1 | 49.2 |
| disjoint median (°) | 1.64 | 5.55 | 1.63 | 4.76 |
| shared / disjoint | 10.5× | 9.6× | 9.3× | 10.3× |
| P(shared > 21°) | 0.306 | 0.997 | 0.179 | 0.998 |
| Jaccard bins (°) | 7.7 / 11.7 / 15.8 / 21.1 | 41.2 / 45.6 / 50.9 / 59.7 | 7.7 / 11.9 / 14.7 / 18.4 | 43.4 / 45.9 / 48.5 / 53.3 |
| shared same-rank / cross-rank (°) | 17.2 / – | 53.5 / – | 17.2 / 14.8 | 54.2 / 48.5 |
| p99.9 vs null B | 16.6 vs 11.9 (1.4×) | 52.6 vs 26.3 (2.0×) | 14.8 vs 11.4 (1.3×) | 48.7 vs 23.2 (2.1×) |

Reading. The shared-cue alignment and its Jaccard grading are present at
initialization: the architecture puts the cue embeddings into the residual. The
ratio of shared-cue to cue-disjoint deviation is ~10× before and after training.
What training changes is the code's dimensionality, PR 383 → 48, and every
angle scales up with it. So SGD does not *build* a shared context code; it
inherits the alignment from the input and preserves it, proportionally, while
compressing the code eightfold. The rank component is real and also inherited:
same-rank shared pairs sit ~5° above cross-rank ones at init and after training,
which is the gap between the main (55.4°) and rank-mixed (50°) corpora.

Consequences for the paper. (1) "SGD builds a shared context code" is retired;
the surviving positive claim is compression: SGD spends the width on accuracy
and shrinks the sense code to ~1% of d, spreading nothing toward orthogonality.
(2) The tail is described as inherited cue-embedding structure that training
preserves and pricing does not remove. (3) Title: "Fifty Dimensions for Ten
Thousand Senses: How a Transformer Packs More Senses than Width" (a third
round objected that "Compression, Not Orthogonality" poses a dichotomy the
introduction itself disowns). (4) Two further reviewer
points fixed in the same pass: the block-cluster bootstrap over-weights
within-block pairs 2× (E[m²]=2) and is replaced by a delete-one-block jackknife;
the registered ±4° violation endpoint (T3: 0.62 vs 0.00) goes back into the
capacity table. (5) The rank-mixed rule's part (b) is labelled a post-hoc
re-scoring in the ledger, not a pass.

### Low-width sweep (registered 2026-09-05, before any run)

Motivation. Liu, Liu & Gore (2025) describe two regimes in the toy autoencoder:
weak superposition (top-m features kept orthogonal, the rest dropped; code as
wide as d) and strong superposition (everything kept, directions overlap
throughout the space). Our d=512–4096 runs are neither: everything is kept,
squashed into PR ≈ 0.01–0.025·d, rare senses degraded but not dropped. At
d=4096 the model can afford that (acc 0.91). The question is what gives when
it cannot.

Runs. `bigrun.py seed={0,1,2} prec=bf16 d={64,128,256}`, γ=0, same corpus,
same 6000 steps, same lr 1e-4 as the registered sweep. Endpoints per run: acc,
per-rank acc, PR of protocol centroids, PR/d, cross-word p50 and p99.9 vs null
B, shared-cue vs disjoint median, same-word raw cosine, held-out loss.

Predictions, written before running:
- (A) Spread. PR/d rises toward O(1) as d falls, p50 approaches null A (the
  isotropic code), and accuracy degrades gracefully. Reading: the model enters
  strong superposition once compression can no longer pay for accuracy; the
  d at which PR/d crosses 0.25 is the transition width.
- (B) Squash. PR/d stays ≤ 0.05 and accuracy collapses toward the prior-only
  ceiling (0.438) or the model diverges. Reading: the low-rank bias beats the
  task even when the task is losing; no superposition regime is reached in
  this architecture.
- (C) Drop. Accuracy collapses on r4–r5 only while r1–r2 hold and PR/d stays
  small: the weak-superposition behaviour (keep the frequent, drop the rare)
  without the orthogonality.
Decision rule: classify each width by PR/d (≥0.25 → A) and by the r1 vs r5
accuracy gap relative to d=4096 (r5 falls to chance-level 0.083 while r1 stays
within 0.05 → C); otherwise B. Report all three seeds; a width whose seeds
disagree is reported as mixed.

Learning-rate caveat, stated in advance. lr 1e-4 was chosen at d=4096 after
divergence probes. If a run at d ≤ 128 diverges (loss NaN or > initial after
1000 steps) or fails to move (acc within 0.02 of prior-only at 6000 steps), it
is rerun once at the toy-arm rate 1e-3 and both are logged; the rerun counts.
Nothing else is retuned.

**Outcome (runs 2026-09-05; `results/big_d{64,128,256}_s*.log`,
`*_lr1e-3.log`, `nulls_d64_lr1e-3.log`, `nulls_d256_lr1e-3.log`, T9).**

At lr 1e-4 (the registered rate) d=256 reached acc 0.453–0.472, within 0.02 of
the prior-only ceiling for seed 0, and d=128 / d=64 did not learn (acc ≤ 0.02,
loss 7.5–8.5). The caveat fired; all nine runs were redone at 1e-3 (seeds 1–2
of d=256 did not trigger on their own and were rerun for comparability; both
sets are in T9). At 1e-3:

| d | acc (3 seeds) | r5 acc | PR protocol (within-word) | PR/d | uncentered PR | word-centroid PR | word |cos| median |
|---|---|---|---|---|---|---|---|
| 64 | 0.872–0.877 | 0.70–0.73 | 8 | 0.12–0.13 | 13 | 11 | 0.21 |
| 128 | 0.908–0.924 | 0.76–0.80 | 10–11 | 0.08 | – | – | – |
| 256 | 0.910–0.935 | 0.72–0.82 | 16 | 0.06 | 30 | 28 | 0.13 |
| 4096 (lr 1e-4, s0) | 0.908 | 0.665 | 50 | 0.012 | 80 | 110 | 0.24 |

Classification by the registered rule: every width is B by the PR/d clause,
but B's second half (accuracy collapses) is false. None of A/B/C as written
occurred. The model squashes and succeeds: 10^4 senses decode at 0.87 in
d=64 with a within-word code of PR 8 and a full sense code of PR 13.

Two consequences, both stated in the paper:
1. The registered width sweep (d=512–4096 at lr 1e-4) was learning-rate
   limited. d=128 at 1e-3 outperforms d=4096 at 1e-4 (0.92 vs 0.91). "Width is
   spent on accuracy" and "capacity is binding" are withdrawn. What the sweep
   shows is that the task needs about a dozen dimensions and SGD gives it
   about that at every width; PR 8–16 at d=64–256 well trained, 50 at d=4096.
   The d^0.65 growth of PR is consistent with centroid-noise inflation (more
   dimensions of noise, same sample size per sense), which the third review
   round predicted; it is not measured directly.
2. The per-word-centered object the paper measured is the within-word (rank
   / cue-pool) code, which given the word needs only ~log 5 dimensions; a
   small PR for it is unsurprising once said aloud. The superposition-relevant
   object is the uncentered sense code: 10^4 sense centroids with PR 13 in
   d=64, 2000 word centroids with PR 11 and median |cos| 0.21, decoded at
   0.87. That is S/d = 156 and S/PR ≈ 770.

Not yet done: per-width learning-rate tuning for d ≥ 512 (the 4096 runs at
1e-3 diverged in the original probes; a μP-style sweep is the proper fix),
shared-cue / disjoint analysis of the uncentered code, and the reviewers'
noise-corrected PR. The paper cannot keep "Fifty Dimensions" as its headline
number without the noise correction; the defensible number is the d=64 result.

### Why does PR grow with width? Rival hypotheses (registered 2026-09-05, before running)

Method: the `hypothesis-generation` and `scientific-critical-thinking` procedures
from K-Dense scientific-agent-skills (Kassis et al. 2026), applied to the
observation above.

Observation, frozen. Protocol (per-word-centered) centroid PR is 8 / 10–11 / 16
at d = 64 / 128 / 256 (lr 1e-3, 3 seeds) and 13 / 19 / 28 / 50 at d = 512 /
1024 / 2048 / 4096 (lr 1e-4, 1 seed). Accuracy 0.87–0.93 at d ≤ 256 (lr 1e-3),
0.72–0.91 at d ≥ 512 (lr 1e-4). Each sense centroid averages 5–48 training
sentences (r5 ≈ 4). Exploratory; the pattern was noticed after viewing results.

Question. Is the width dependence of PR a property of the learned code, or of
the measurement / training protocol?

Rivals (all `candidate`):
- H1 mechanism: SGD uses more dimensions for the sense code when more are
  available.
- H2 measurement artifact: centroid sampling noise occupies ~d dimensions at
  fixed sentences per sense, so PR of noisy centroids rises with d while the
  signal PR does not.
- H3 training confound: lr 1e-4 at d ≥ 512 and lr 1e-3 at d ≤ 256 are
  different training regimes; PR and accuracy are not comparable across the
  lr boundary.
- H4 wrong object: per-word centering leaves the within-word code, which given
  the word needs ~log 5 dimensions at any width; the width-relevant object is
  the uncentered code.

Discriminating tests and predictions:
1. Split-half signal PR. Split each sense's sentences into two random halves
   A, B; Σ_signal = ½(Cov(A,B) + Cov(B,A)) is unbiased for the signal
   covariance (noise is independent across halves); Σ_noise = ½ Cov(A−B).
   Report PR(Σ_signal), PR(Σ_noise), and the raw PR for d = 64, 256 (lr 1e-3)
   and 512, 4096 (lr 1e-4), seed 0.
   H2 predicts PR(Σ_signal) roughly flat (≈ 8–16) across d while raw PR grows;
   H1 predicts PR(Σ_signal) grows with d in step with raw PR. Indeterminate if
   the split-half estimate is unstable (r5 has ≈ 4 sentences; report with and
   without r5).
2. Learning-rate confound. Run d = 512, 1024, 2048 at lr 1e-3 and d = 4096 at
   lr 3e-4 (1e-3 diverged in the original probes), seed 0, same steps.
   H3 predicts accuracy at d ≥ 512 rises to ≥ 0.92 and PR changes; if so the
   registered sweep is reported as lr-limited and only best-of-lr rows are
   compared. If accuracy does not change, lr is not the confound and the
   d ≥ 512 rows stand.
3. Uncentered PR across widths (already measured at 64 / 256 / 4096: 13 / 30 /
   80) is reported alongside as the H4 object; split-half applied to it too.

Decision rule. If test 1 gives PR(Σ_signal) within a factor 1.5 across all
four widths, the paper reports the code as ~10–20 dimensional at every width,
with raw PR labelled noise-inflated, and drops "fifty" from the title. If
PR(Σ_signal) grows by ≥ 3× from 64 to 4096, H1 stands and the title number is
the corrected d = 4096 value. Test 2 decides whether the d ≥ 512 accuracies
appear in the paper at all.

**Outcomes (2026-09-05; `splithalf.py`, `results/splithalf_s0.log`;
`results/big_d{512,1024,2048}_s0_lr1e-3.log`, `big_d4096_s0_lr3e-4.log`).**

Test 1, split-half signal PR (seed 0; 1401 words have every sense in both
halves; median 3 sentences per sense per half):

| d (lr) | protocol PR raw / signal / noise | uncentered raw / signal | word centroids raw / signal | signal share of trace (protocol) |
|---|---|---|---|---|
| 64 (1e-3) | 7.7 / 7.4 / 10.1 | 12.3 / 12.2 | 11.1 / 11.1 | 0.89 |
| 256 (1e-3) | 16.2 / 15.5 / 21.0 | 28.9 / 29.2 | 28.0 / 27.6 | 0.87 |
| 512 (1e-4) | 13.9 / 12.6 / 15.9 | 15.5 / 15.4 | 15.2 / 15.2 | 0.68 |
| 4096 (1e-4) | 49.2 / 42.1 / 61.1 | 76.4 / 78.3 | 102.5 / 103.8 | 0.68 |

Re-run on the best-of-lr checkpoint per width (protocol code only,
`results/splithalf_s0.log`): raw / signal PR = 7.7/7.4 (64), 10.0/9.8 (128),
16.2/15.5 (256), 24.6/22.9 (512, lr 1e-3), 25.2/32.7 (1024, lr 1e-3, noisy),
29.8/26.1 (2048, lr 1e-4), 49.2/42.1 (4096). Monotone except the two widths
without a near-optimal rate.

Signal PR rises 7.4 → 42 from d=64 to 4096 (5.7×, above the registered 3×
threshold). H2 (noise inflation) is rejected as the main explanation: noise is
11–32% of the trace and the corrected d=4096 value is 42, not a dozen. H1
stands: SGD uses more dimensions when more are available. The title number is
the corrected value, "about forty" rather than fifty; the paper reports both.
Dropping r5 changes nothing (r1–r4 rows within 0.5 of the all-rank rows).

Test 2, learning-rate confound (seed 0, same steps):

| d | acc at 1e-4 | acc at 1e-3 | acc at 3e-4 | best |
|---|---|---|---|---|
| 64 | 0.000 | 0.872 | – | 0.872 |
| 128 | 0.004 | 0.924 | – | 0.924 |
| 256 | 0.453 | 0.921 | – | 0.921 |
| 512 | 0.722 | 0.902 | – | 0.902 |
| 1024 | 0.766 | 0.815 | – | 0.815 |
| 2048 | 0.854 | 0.312 | – | 0.854 |
| 4096 | 0.908 | diverged (2026-08 probes) | 0.867 | 0.908 |

H3 confirmed. The optimal rate falls with width, roughly as 1/d, and the
registered sweep's accuracy trend (0.72 → 0.91) was the fixed rate, not
capacity. Best-of-lr accuracy is 0.87 at d=64 and 0.90–0.93 at every width
from 128 up, with 1024 the one width where neither rate was near optimal.
Combined with test 1: width buys dimensions (signal PR 7 → 42) and not
accuracy (flat at ~0.9). The task is solved at d=128 with a ~10-dimensional
within-word code; at d=4096 SGD spends ~40 on the same task. This is the
paper's width result now. "Capacity is binding" is withdrawn.

Per-width lr tuning beyond two points was not done; the d=1024 row is the
visible cost. A μP-style sweep is the proper fix and is listed as future work.

## Planned work (registered 2026-08-24; perf item 1 since landed, see below)

Written off-box (Ryzen 1200 dev machine, no GPU); every number below is
static arithmetic from `harness.train`/`model.py` shapes, NOT measured.
All three items need a verification run on the 5060 before they count.

### Step-cost accounting (corrects an earlier working assumption)

At arm-5 shape (d=4096, V=14082, bs=256, T=14 → 3584 tokens, 3328 loss
positions), FLOPs per training step (2 FLOP/MAC):

| part | TFLOP/step |
|---|---|
| block fwd (24d²/token) | 1.44 |
| block bwd | 2.89 |
| block recompute (activation checkpointing) | 1.44 |
| head fwd (2dV/position) | 0.38 |
| head bwd | 0.77 |
| total | 6.92 |

The BLOCK is 83% of the step; the unembedding head is 17% — the opposite
of the intuition that a V-wide head dominates. 6.92 TFLOP / 0.435 s =
15.9 TFLOPS achieved, i.e. at consumer-Blackwell TF32-dense peak: the run
is compute-bound, not launch-bound. Consequences: (a) any precision win
must cover the block, not just the head; (b) CUDA graphs / torch.compile
are not worth pursuing here.

### Perf item 1 — drop block checkpointing — DONE (2026-08-24)

`TinyLM(gc=True)` recomputes the block in backward. That recompute is
21% of a compute-bound step, and it was adopted for
the CE-head memory transient (Amendment 3), not for the block itself.
Storing block activations instead costs ~0.82GB (qkv 176MB, MLP hidden
pre/post-ReLU 2×235MB, three d-wide tensors ~176MB) against 1.6GB
measured headroom. Recompute-vs-store is bitwise identical here (no
dropout, no RNG in Block), so this is a pure engineering change with no
statistics impact and no new registered claim.
Expected: 43 → ~35 min train. GATE: re-run the VRAM preflight; if peak
exceeds the 6.5GB train budget, revert and keep checkpointing.

LANDED together with the head-gather change (Amendment 4): measured peak
6.03GB allocated / 6.34GB reserved — inside this gate — and 0.291 s/step,
i.e. ~29 min train, past the ~35 min projection because the head fix also
landed. Bitwise-exactness was verified on the gradient fingerprint
(delta exactly 0.0).

### Perf item 2 — bf16 autocast — DONE, registered as Amendment 5

Autocast the whole forward (block + head) with fp32 master weights;
bf16 tensor throughput is ~2× TF32 on this hardware, so the realistic
ceiling is 1.5-1.7× on the GEMM-bound step → ~30 → ~20-23 min.
bf16 round-to-nearest is ~2e-3 relative vs TF32's ~5e-4 — a 4× wider
envelope than Amendment 2 claimed, so this does NOT inherit that
amendment's blessing. Requires, before any science run:
1. loss and max-grad rel-delta vs the current TF32 path at arm-5 shape
   (same protocol as the Amendment 3 verification: 7.7e-8 / 2.8e-7);
2. split-half reliability on a bf16-trained seed 0 within noise of the
   TF32 run's 0.88/0.78/0.67/0.56/0.48;
3. λ penalty statistics (`F.normalize` + `H @ H.T`) forced to fp32 —
   cos² in bf16 is not acceptable for a term that shapes geometry.
If (2) fails, bf16 is rejected and ~30 min is the floor.

LANDED: both pre-registered gates measured and passed (see Amendment 5);
the fp32 penalty-statistics requirement is enforced in `harness.train`.

### NVFP4: rejected as a training format, registered as an analysis arm

Checked (2026-08-24, vendor + community sources): the 5060 is `sm_120`
with native NVFP4 (E2M1 + E4M3 per-16-element block scale + per-tensor
scale) on 5th-gen tensor cores, but cuBLASLt does not expose NVFP4 matmul
on `sm_120` (CUTLASS `mma.sync.aligned.block_scale` hand-kernels are the
community path), Transformer Engine targets `sm_100`, and torchao's
NVFP4 is prototype with M%64/N%128 constraints.

Rejected for training on protocol grounds, not just plumbing: NVIDIA's
own NVFP4 pretraining recipe requires 16×16 Random Hadamard Transforms on
Wgrad inputs, stochastic rounding of gradients, 2D block scaling, and
~15% of layers held in BF16/FP32 (typically the final ones). This model
is one block plus an unembedding — there is no depth to hide low
precision in, and the layers the recipe protects are essentially the whole
model. RHT + stochastic rounding also alter gradient-noise structure,
which is an intervention on the dependent variable (SGD's choice of
packing), not a numerics footnote.

Registered instead — **post-hoc quantization arm** (`quant.py`, to write):
simulate NVFP4 in fp32 (E2M1 grid, 16-element blocks, E4M3 block scales,
per-tensor global scale) plus plain round-to-nearest at 8/4/3/2 bits, then
re-run the EXISTING analysis on already-trained checkpoints. No kernels,
no hardware dependency, CPU-runnable, deterministic.
- Inputs: `results/big_g0.0_l0.0_s{0,1,2}.ckpt` and `big_g0.0_l1.0_s0.ckpt`
  (λ=0 vs λ=1 is the contrast that matters).
- Endpoints, all pre-existing: held-out per-rank acc, same-word median cos,
  weighted cross-word interference, `packing_report` deviation quantiles,
  and the shared-cue / disjoint / co-occurring split from `posthoc.py`.
- Pre-registered predictions: (i) behavior survives to lower bit-widths
  than geometry does; (ii) damage lands first on the thin-margin
  shared-cue population; (iii) the λ=1 checkpoint tolerates ≥1 bit-width
  lower than λ=0 at matched accuracy. Prediction (iii) is the one that
  would make "priced interference buys quantization headroom" more than a
  story; failing it kills that thesis cleanly.
- Order: this arm is INDEPENDENT of the uniform-penalty run and needs no
  GPU, so it can proceed off-box while the 5060 is busy.

### Queue order

1. Uniform-weighted cross-word penalty — DONE, see Results: alignment is
   load-bearing; uniform pricing adds only marginal disjoint-tail
   tightening.
2. λ=3 dose at capacity.
3. Perf item 2 (bf16) only if its three gates pass — PASSED, landed as
   Amendment 5; with Perf item 1 this takes an arm-5 run from 45 to ~18
   min.
4. Post-hoc quantization arm (off-box, any time).

---

# Toy arm (d=64, capacity slack) — protocol, results, caveats

Moved here from HANDOFF.md on 2026-09-02 when that file was retired; the
toy runs are the 52 logs `results/grid_*.log` and `results/beh_*.log`,
5 seeds per configuration, 2500 steps, CPU. Every number below has its
seed range in `results/TABLES.md` (T1, T2).

## Headline results (toy arm)

1. **Same-word senses pack negatively, always.** All healthy runs classify
   SIMPLEX. At γ=0, d=64, mean rank-pair cosines sit on the regular-simplex
   value −1/(k−1) = −0.333 almost exactly (six pairs: −0.28…−0.39).
   Mutually-exclusive features taking antipodal/simplex geometry costs the
   loss nothing, exactly as superposition theory predicts.
2. **γ (weight decay) is graded in a window, then kills.** 5 seeds per γ,
   d=64, 2500 steps:
   - γ ≤ 1: indistinguishable from 0 (acc 0.91, interference ~0.33-0.36).
   - γ ∈ [1.5, 3]: graded reallocation — acc holds (0.908 → 0.875),
     interference rises monotonically 0.33 → 0.40, resid median softens
     −0.35 → −0.23.
   - γ = 4: death onset (acc 0.814±0.071, split-half reliability floor
     collapses to 0.386 → geometry numbers there are untrustworthy, per
     protocol). γ = 5: full death (earlier sweep).
   Note the inversion: interference *rises* under decay — decay eats
   weakly-supported directions first, leaving centroids dominated by shared
   global components.
3. **Frequency-dependent allocation CONFIRMED, and dose-responsive in γ.**
   The two rarest ranks (r3-r4) merge while every other pair stays
   simplex-negative. Across 30 runs (5 seeds × γ ∈ {0,1,1.5,2,3,4}):
   - r3-r4 is the least-negative of the 6 pairs in 26/30 runs (23/25 in the
     healthy regime γ ≤ 3; `stats.py` T1 is the authoritative count) (null 1/6;
     binomial p ≈ 8×10⁻⁷ already at 9/10 in the γ∈{0,2} grid alone).
   - Monotone dose response, healthy regime: r3-r4 mean cos −0.08 (γ=0)
     → −0.11 (1) → −0.01 (1.5) → +0.07 (2) → +0.16 (3), while the
     other-pairs mean goes the *opposite* way, −0.37 → −0.42. Decay merges
     the rarest senses first and sharpens frequent-sense packing.
   - Split-half reliability ≥ 0.93 everywhere healthy, so this is geometry,
     not centroid noise.
   - Solution families still vary by seed: γ=2 s=2 found two antipodal
     pairs (r1-r4 = −0.97, r2-r3 = −0.94) with r3-r4 merged at +0.37.
4. **Targeted intervention dominates γ on every endpoint.** Cross-word-only
   overlap penalty (λ · mean cos² between resid vectors at prediction
   positions whose targets are cont tokens of different words; same-word
   pairs excluded; co-activation weighting implicit in batch sampling).
   `harness.train(lam=…)`, `run.py lam=…`; mask logic guarded by
   `test_penalty.py`. 5 seeds per λ, d=64, 2500 steps:
   - Interference: 0.330 (γ=0) → 0.244 (λ=1) → 0.197 (λ=3). γ could only
     *raise* it (0.40 at γ=2-3).
   - Accuracy: 0.909–0.912 at every λ ∈ {0.3,1,3,10} — zero cost, where
     γ=3 paid 4 points.
   - **The rare-rank merge is abolished, not softened**: at λ=1, r3-r4
     minus other-pairs mean = −0.009 (max-pair in 0/5 runs, vs 4/5 at
     baseline). The merge is therefore an allocation *choice* made when
     cross-word interference is unpriced — price it and the model finds
     full per-word simplex + lower interference + same accuracy.
   - λ=10 over-penalizes: asymmetry returns (+0.19), reliability drifts
     to 0.819. Sweet spot λ ∈ [1, 3]. Caveat: min split-half reliability
     declines with λ (0.93 → 0.83) — watch it in any follow-up.
5. **The merge has a behavioral price, and λ doesn't pay it.** Per-rank
   test accuracy/loss (`by_rank` from `harness.capture`), 5 seeds per arm,
   compared at MATCHED frequent-sense (r1) loss:
   - λ=1 vs γ=2 (r1 loss 0.244 vs 0.226): rare acc r3/r4 0.897/0.896 vs
     0.854/0.851.
   - λ=3 vs γ=3 (r1 loss 0.256 vs 0.263): rare acc 0.898/0.891 vs
     0.810/0.781; rare r4 loss 0.457 vs 0.645.
   - γ degrades rare senses disproportionately (rare/frequent loss ratio
     1.93 → 2.01 → 2.45 across γ = 0→2→3); λ holds rare-sense behavior at
     baseline. Geometry↔behavior link closed: the rare-rank merge costs
     rare-sense accuracy specifically; pricing cross-word interference
     avoids both.
   - λ raises frequent-sense loss slightly (0.20 → 0.26) with no accuracy
     change — it eats logit margins, not decisions.
6. Task solvability: cont-token accuracy ≈ 0.91 at γ ≤ 1 (chance 1/48 =
   0.02; prior-only ceiling 0.48). CORRECTED 2026-09-02: this IS saturation.
   `check_gen.py toy` shows 17% of toy sentences contain a distractor cue
   that forms a valid pair with the target word for a different
   continuation; with no positional code the prefix is a bag, so a naive
   order-blind reader tops out at 0.917 overall. Longer schedules cannot
   lift it. See caveat below.

## Registered protocol decisions (do not silently change)

- No LayerNorm anywhere: keeps AdamW decay semantics identical to the
  validated knob (LN would turn γ into effective-LR reparametrization).
- Untied embeddings: token-identity geometry must not leak into the
  unembedding space contaminating sense measurements.
- ReLU MLP included (the cue⊕target conjunction is a nonlinear AND);
  single attention head; no positional embeddings (order carries no signal).
- Loss masked to continuation positions only (cue/filler entropy is pure
  gradient noise for this question).
- ≥5 seeds per config: solution families differ seed-to-seed (observed:
  one seed flipped r3-r4 to positive while five other pairs stayed
  simplex-negative). Medians over single seeds are meaningless here.
- Readout sites: attention output pre-residual AND final residual at the
  prediction position (both captured; resid is primary). At 1 layer the
  "PS-Eval mid-stack attention placement" guidance does NOT transfer —
  placement becomes a variable only at 2+ layers. Do not cite our setup as
  replicating that finding.
- Angle measurements: per-word centering before centroids; raw cosines are
  invalid (anisotropy + shared token identity swamp sense components).
- ε-dial null prediction at toy width stands: capacity ≈ d below
  ε*(d) ≈ ±31° at d=64, so tolerance-widening must do nothing there. If it
  appears to, suspect a bug first.

## Known caveats

- **Order-blind conflicts (found 2026-09-02, `check_gen.py`).** The model
  has no positional code and one causal layer, so its output at the target
  word depends only on the multiset of prefix tokens. A distractor cue that
  also pairs with the target word (same-group collision) is unresolvable by
  cue identity alone. Toy corpus (12 words, 3 blocks): 17% of sentences,
  rank-dependent (r1 13%, r4 21%); naive best-guess ceiling 0.917 overall,
  0.79 for r4. Observed toy acc 0.91 sits at that ceiling, and per-rank
  toy behavior (headline 5) carries this rank-dependent load — the model
  evidently resolves some collisions via the distractor word also being in
  the bag, since r4 acc 0.89–0.91 exceeds the naive 0.79, so treat the toy
  per-rank numbers as relative between arms, not absolute. Capacity arm
  (2000 words, 500 blocks): 0.62% of sentences, ceiling 0.993–1.0 per rank
  — negligible; arm-5 behavioral endpoints are clean. Also: 4 of 814 r5
  test rows have a sense with zero train examples (reachable ceiling
  0.995); 9 of 9999 senses have <2 train examples.
- Corpus-level `freq_err = 0.0000` is exact by construction (fixed per-rank
  budgets per word); real Zipf variation enters through distractor sampling
  and what the *model* sees, which the co-activation matrix measurement
  picks up empirically.
- Cue sets saturate pools at current sentences/pool ratio → the blocks
  regime is a binary sharing contrast. Graded overlap needs pool-size
  tuning; earn that complexity only after the binary effect is published-
  grade solid.
- Held-out measurement uses the 10% test split; models have seen 90%.
  Fine for geometry comparisons across γ (uniform bias), not for absolute
  claims about novel-context generalization.
- `emb` stream is vacuous by construction (one row per type) — kept in
  output only as a negative-control site.
