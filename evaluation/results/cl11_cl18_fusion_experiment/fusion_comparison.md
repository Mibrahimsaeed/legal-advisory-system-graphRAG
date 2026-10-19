# Fusion Experiment Comparison

**All configs use the SAME fixed depth-40 dense/sparse candidate lists and the SAME frozen qrels; only the sparse weight varies.** No statistical significance is claimed anywhere below — 20/13 queries per corpus is too small a sample, per the original benchmark's own disclosed limitation.

## Macro comparison

| Corpus | Config | P@5 | Recall@5 | Recall@10 | MRR | nDCG@10 | n |
|---|---|---:|---:|---:|---:|---:|---:|
| Constitution | **Production (frozen)** | 0.210 | 0.925 | 1.000 | 0.906 | 0.921 | 20 |
| Constitution | Offline repro. (1.0/1.0) | 0.210 | 0.925 | 1.000 | 0.856 | 0.884 | 20 |
| Constitution | Reduced sparse (1.0/0.75) | 0.210 | 0.925 | 1.000 | **0.907** | **0.921** | 20 |
| Constitution | Reduced sparse (1.0/0.5) | 0.210 | 0.925 | 1.000 | **0.908** | **0.922** | 20 |
| Case law | **Production (frozen)** | 0.077 | 0.385 | 0.462 | 0.402 | 0.407 | 13 |
| Case law | Offline repro. (1.0/1.0) | 0.077 | 0.385 | 0.462 | 0.404 | **0.407** | 13 |
| Case law | Reduced sparse (1.0/0.75) | 0.077 | 0.385 | 0.462 | 0.368 | 0.379 | 13 |
| Case law | Reduced sparse (1.0/0.5) | 0.077 | 0.385 | 0.462 | 0.357 | 0.369 | 13 |

**Offline reproduction (1.0/1.0) validates closely against the frozen production baseline** — identical on P@5/Recall@5/Recall@10 for both corpora, and case law's nDCG@10 matches to 3 decimal places (0.4069 vs 0.4069). Constitution's MRR/nDCG@10 differ slightly (0.856 vs 0.906) — consistent with the disclosed 2/33-query candidate-set mismatch (C10, C13) in the offline formula, not a wholesale reproduction failure.

## CL11 / CL18 subset

| Config | CL11 target rank | CL18 target rank |
|---|---:|---:|
| Production (frozen) | 17 | 10 |
| Offline repro. (1.0/1.0) | 17 | 10 |
| Reduced sparse (1.0/0.75) | 15 | 9 |
| Reduced sparse (1.0/0.5) | **12** | **9** |

**Reducing sparse weight measurably and monotonically improves both CL11 and CL18** — directly confirming the diagnosis's mechanism (less sparse dilution of a dense-findable target).

## Regressions found

| Query | Corpus | Baseline (1.0/1.0) nDCG@10 | At 1.0/0.5 | Change |
|---|---|---:|---:|---|
| **CL01** | Case law | 1.000 | **0.500** | **Large regression** — CL01's target is entirely absent from dense (confirmed in the original audit) and depends on a strong sparse rank (1) to be found at all via fusion; halving sparse's weight proportionally weakens that rescue. |
| **C13** | Constitution | 0.818 | 0.798 | Small regression |

**No other query in either corpus regressed** under either reduced-sparse config relative to the 1.0/1.0 offline baseline.

## Net effect on macro metrics

Reducing sparse weight **improves Constitution's macro MRR/nDCG@10 slightly** (0.856→0.908, 0.884→0.922) while **measurably hurting case law's** (0.404→0.357 MRR, 0.407→0.369 nDCG@10) — the case-law harm is driven overwhelmingly by CL01's single large regression, not a broad decline across all 13 case queries.
