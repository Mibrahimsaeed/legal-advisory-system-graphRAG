# Blind validation — 40-case sample

> **Scope.** These figures describe a **40-case blind validation sample** drawn at random (seed `20260927`) from the 164 cases that decision run `dec_pilot` marked `auto_accepted`.
>
> **They are NOT the accuracy of the full 187-case corpus**, and not the accuracy of the pipeline as a whole. The 23 `needs_review` cases were excluded by construction, so the hardest cases the classifier itself flagged are absent from this measurement.

Generated 2026-09-26T22:49:37+00:00 · decision run `dec_pilot` · matched strictly by `doc_id`.

## Matching

| | |
|---|---|
| doc_ids in blind CSV | 40 |
| machine classifications found in `dec_pilot` | 40 |
| human labels present and valid | 40 |
| **matched pairs evaluated** | **40** |
| unmatched (no machine row) | 0 |
| invalid human label | 0 |

## Headline

* **Overall accuracy: 100.0%** (40/40 agree)
* **Macro F1: 1.000** (measurable classes only)
* Micro F1: 1.000 · Weighted F1: 1.000
* **Disagreements: 0**

## Label distribution

| domain | human (ground truth) | machine (`dec_pilot`) |
|---|---:|---:|
| `family_law` | 38 | 38 |
| `criminal_law` | 2 | 2 |
| `other_uncertain` | 0 | 0 |

## Confusion matrix

Rows = human ground truth, columns = machine prediction.

```
gold \ predicted        family_law     criminal_law  other_uncertain
--------------------------------------------------------------------
family_law                      38                0                0
criminal_law                     0                2                0
other_uncertain                  0                0                0
```

## Per-class metrics

| domain | precision | recall | F1 | support (human) | predicted (machine) |
|---|---:|---:|---:|---:|---:|
| `family_law` | 1.000 | 1.000 | 1.000 | 38 | 38 |
| `criminal_law` | 1.000 | 1.000 | 1.000 | 2 | 2 |
| `other_uncertain` | n/a | n/a | n/a | 0 | 0 |

`n/a` means the value is **undefined, not zero** — a class with no human examples in this sample cannot have a recall, and one the machine never predicted cannot have a precision. Undefined classes are excluded from the macro average rather than counted as 0.

## Disagreements

None — the machine and the reviewer agreed on all 40 cases.

## Full case-by-case result

| # | doc_id | source_relpath | machine | human | agree |
|---:|---|---|---|---|:--:|
| 1 | `0513f7c1139b729f4cccf83d` | `1983K604` | family_law | family_law | ✓ |
| 2 | `05c9109a43646f6748f7a76b` | `1982L560` | family_law | family_law | ✓ |
| 3 | `100fcea4a7af3ece34f84bf4` | `1976L141` | family_law | family_law | ✓ |
| 4 | `124ce29725d37ca50d0df611` | `1985L59` | family_law | family_law | ✓ |
| 5 | `15d32c0f18ba1c7dfb284274` | `1983S965` | family_law | family_law | ✓ |
| 6 | `15e0969ca0f0fa564c597791` | `1978L99` | family_law | family_law | ✓ |
| 7 | `1782fdc21be1d4739c9f2cce` | `1985L266` | family_law | family_law | ✓ |
| 8 | `1cc87fb6395d59796b56284c` | `1982P32` | family_law | family_law | ✓ |
| 9 | `211a8f204d4ed5b2c99b9923` | `1968L111` | family_law | family_law | ✓ |
| 10 | `22f49ac397f6f24ebaa54c55` | `1986K77` | family_law | family_law | ✓ |
| 11 | `23a39833e41c2d3bbc105b2a` | `1984K265` | family_law | family_law | ✓ |
| 12 | `23cdc5265254eb0a1293d1c5` | `1986L382` | family_law | family_law | ✓ |
| 13 | `2670965d45ebe56747d554ff` | `1978P9` | family_law | family_law | ✓ |
| 14 | `27fa9958672a5f2602e56b1b` | `1986K527` | family_law | family_law | ✓ |
| 15 | `3e0aab5b37cdf0be775c7e12` | `1975L114` | family_law | family_law | ✓ |
| 16 | `44bca0a557b84b71f89662a1` | `1974L10` | family_law | family_law | ✓ |
| 17 | `45c247d7ef78950725beb99f` | `1985Q38` | family_law | family_law | ✓ |
| 18 | `5114836790107329161990eb` | `1984L32` | family_law | family_law | ✓ |
| 19 | `546d8b11944203c4c61ade1f` | `1979L95` | family_law | family_law | ✓ |
| 20 | `5961df8e1ddaeba020c8e16b` | `1975L66` | family_law | family_law | ✓ |
| 21 | `62135a109dab3dcdb0eff2e1` | `1969L108` | family_law | family_law | ✓ |
| 22 | `6a31e04254fd0657a63197c8` | `1985L429` | family_law | family_law | ✓ |
| 23 | `6c437bc05b51b44aef8631ef` | `1978L14` | family_law | family_law | ✓ |
| 24 | `6ea7475991392596a155e52f` | `1984L2539` | family_law | family_law | ✓ |
| 25 | `7d5671889dee8117c2f930a3` | `1977K78` | family_law | family_law | ✓ |
| 26 | `871c4c3395a6882ef3ad9e9a` | `1986K255` | family_law | family_law | ✓ |
| 27 | `96112ecbe3dc83b3a5597c7e` | `1985L2560` | family_law | family_law | ✓ |
| 28 | `9d11ae91045bf6356fd26f3a` | `1971L118` | family_law | family_law | ✓ |
| 29 | `a0b746265d8effbeed895012` | `1983L67` | family_law | family_law | ✓ |
| 30 | `a864c52ea66d56dcde0dad3e` | `1967S49` | family_law | family_law | ✓ |
| 31 | `b23703da404594ae71dbd343` | `1968S1178` | criminal_law | criminal_law | ✓ |
| 32 | `c0e9bcc1db2474205324e109` | `1986L443` | family_law | family_law | ✓ |
| 33 | `c5053208d7c07c55e42a4e62` | `1985L2836` | family_law | family_law | ✓ |
| 34 | `c927036e2cfb8d0be8d42b74` | `1983K233` | family_law | family_law | ✓ |
| 35 | `d020a081decdd6e0f760d3a7` | `1985K275` | family_law | family_law | ✓ |
| 36 | `d7ab7202ddb4c4e84f06101b` | `1968L3076` | family_law | family_law | ✓ |
| 37 | `e08bc07bb938d64079d031bd` | `1984S18` | criminal_law | criminal_law | ✓ |
| 38 | `f7afd3f61f59d8f9798b8742` | `1981L224` | family_law | family_law | ✓ |
| 39 | `f9d00d5a73cb5f8bf607fef4` | `1985L255` | family_law | family_law | ✓ |
| 40 | `fa04ac4babcc088a7ecaad1b` | `1986L226` | family_law | family_law | ✓ |

## How to read this

* The sample is drawn **only from `auto_accepted`**, so it measures how often the classifier is right *when it is already confident*. It says nothing about the cases it routed to review.
* The corpus itself is family-law-enriched by construction (cases were acquired largely through Family Courts Act 1964 and custody/maintenance queries), so a family-dominated label distribution is expected and a high accuracy on it is easier to achieve than on a balanced corpus.
* With a class of this size, a single disagreement moves accuracy by 2.5%. Treat the per-class figures for any class with a handful of examples as indicative only.
