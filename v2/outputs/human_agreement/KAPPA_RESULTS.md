# Human agreement: Ryan vs the models (first pass, 50 calibration posts)

Generated 2026-10-30 by `score_ryan.py` in this folder (run from `v2/`). Bootstrap: 2,000 resamples of posts, percentile 95% intervals, seed 20261030.

## Source

- Ryan's sheet: `v2/annotation/annotation_sheet_ryan.xlsx`, uploaded by Ryan-yu-wei to `cursor/v2-research-725c` in commit `f98c050` (GitHub web upload, 2026-10-30 18:15 EDT). It is already on this branch, so it is referenced in place, not copied. Ryan's branch `ev_funnel_revision_Ryan` has no new commits since April 2026 and was not touched.
- Model labels: Qwen2.5-14B-instruct, `v2/outputs/llm_labels.csv` (prompt `funnel-v2`). Small model: `v2/annotation/annotation_model_key.csv` (deberta-v3-xsmall zero-shot; stage and aspects only, one merged `ownership` stage, no families).
- No Daisy labels exist yet (`annotator` is only ever `Ryan`), so there is **no human-vs-human kappa**.

## Validation (`validation.csv`)

| Check | Result |
|---|---|
| Rows in sheet / blank sheet | 800 / 800, same id set, no duplicates |
| Title, excerpt, subreddit, calibration flag vs blank (matched by id) | 0 mismatches. Row order differs (Ryan sorted calibration rows to the top). |
| Rows coded | **50 of 800**, all 50 are the `calibration = yes` rows. 0 of the 750 main rows coded. |
| Missing cells in the 50 coded rows | 0 |
| Codes outside the codebook | 0 (all 0/1, stage and stance codes valid) |
| `none` consistent with aspects | yes, 0 conflicts |
| Rows with every family 0 | 9 (allowed by the codebook for joke/no-subject posts) |
| Ids present in 14B labels (ok=True) / small key | 50 / 50 |
| File format | `.xlsx`, not `coder_ryan.csv` as the README asked; read fine. Notes are in Chinese. |

**Caveat:** the codebook says the 50 calibration posts are for Daisy and Ryan to compare before freezing the codebook, and are *left out of kappa*. These numbers are therefore a preliminary read of model quality on a small, non-random set, not the paper's validation result.

## Headline (n = 50 posts, `agreement_scores.csv`)

| Field | Comparison | % agree [95% CI] | Cohen's κ [95% CI] | Krippendorff's α [95% CI] |
|---|---|---|---|---|
| Stage (7 codes) | Ryan vs 14B | 34% [22, 48] | 0.16 [0.02, 0.30] | 0.12 [−0.05, 0.27] |
| Stage, onboarding+LTO merged | Ryan vs 14B | 40% [26, 54] | 0.21 [0.06, 0.36] | 0.18 [−0.01, 0.34] |
| Stage, onboarding+LTO merged | Ryan vs small model | 26% [14, 38] | 0.08 [−0.05, 0.21] | −0.00 [−0.17, 0.15] |
| Family (8 binary × 50 = 400 cells) | Ryan vs 14B | 86% [83, 90] | 0.44 [0.31, 0.56] | 0.44 [0.31, 0.56] |
| Aspect (7 binary × 50 = 350 cells) | Ryan vs 14B | 91% [88, 94] | 0.50 [0.34, 0.65] | 0.50 [0.33, 0.64] |
| Aspect (7 binary × 50 = 350 cells) | Ryan vs small model | 93% [91, 96] | 0.56 [0.38, 0.72] | 0.56 [0.39, 0.72] |
| Stance | Ryan vs 14B | 64% [50, 76] | 0.14 [−0.07, 0.37] | 0.13 [−0.12, 0.35] |

Per family (Ryan vs 14B): charging_energy κ 0.81; battery_range 0.56; reliability_service 0.51; cost_incentives 0.47; dealer_access 0.47; transition_narratives 0.39; policy_market 0.35; purchase_selection 0.18. Intervals are wide (most span ~0 to ~0.8) because positives are rare in 50 posts. `environment` aspect: Ryan never marked it, so κ is undefined/0.

### By audience

| Field | Enthusiast (n=23) κ [CI] | Shopper (n=27) κ [CI] |
|---|---|---|
| Stage, Ryan vs 14B | 0.07 [−0.10, 0.25] | 0.22 [0.04, 0.41] |
| Family pooled, Ryan vs 14B | 0.54 [0.34, 0.71] | 0.37 [0.20, 0.53] |
| Aspect pooled, Ryan vs 14B | 0.68 [0.45, 0.86] | 0.36 [0.15, 0.56] |
| Aspect pooled, Ryan vs small | 0.56 [0.32, 0.75] | 0.56 [0.26, 0.78] |

## Stage confusion, Ryan (rows) vs 14B (columns) — `confusion_stage_ryan_vs_qwen14b.csv`

| Ryan \ 14B | aware | consid | eval | purch | onboard | LTO | unclear |
|---|---|---|---|---|---|---|---|
| consideration | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| evaluation | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| purchase | 0 | 1 | 0 | 2 | 0 | 0 | 1 |
| onboarding | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| long_term_ownership | 0 | 1 | 1 | 0 | 3 | 1 | 3 |
| unclear | 11 | 3 | 4 | 1 | 0 | 0 | 11 |

Ryan: 30/50 unclear, 0 awareness. 14B: 16/50 unclear, 11 awareness.

## Where they disagree most

1. **14B's "awareness" is Ryan's "unclear."** All 11 posts the 14B called awareness, Ryan called unclear. They are news, product reveals, or chatter with no buying journey, which the codebook says is `unclear`. Examples: `qru24x` "Subaru reveals the Solterra SUV…", `1jwfmvp` "Tesla Cybertruck With 350-Mile Range…", `1obmzbo` "Future EVs: Every Electric Vehicle Coming Soon", `uxdk19` "Can somebody tell me where the gas tank is? /s", `2wlzhq` "Does anybody on this subreddit work in the EV industry?". **This matters for the paper:** the shopper "awareness 32.7%" headline share likely includes a lot of news/no-journey posts.
2. **14B pushes stage onto news and reviews.** Ryan-unclear posts that 14B gave consideration/evaluation: `164nx33` Doug DeMuro Ioniq 6 review (14B consideration), `109vpyq` "Jaguar I-Pace catches fire…" (evaluation), `13j9zrf` Model 3 Performance review (evaluation), `7bwfbb` "Senate Tax bill will keep $7500 EV credit" (consideration), `v3bj5w` highway range test (evaluation).
3. **Ownership: 14B says onboarding or unclear where Ryan says long-term.** Ryan's 9 LTO posts: 14B got 1. Examples: `1fm5p99` "Shout out to Brooklyn Service Center" (14B onboarding), `zdg95b` "Charging failures" (onboarding), `1au0rnt` TuxMats floor-mat review (unclear), `1nsmyn1` "iOS App Battery usage changed?" (unclear), `mm5678` "Ever driven off without the keys?" (unclear). The enthusiast "onboarding 26.8%" share may be partly long-term ownership.
4. **Evaluation vs consideration.** All 3 Ryan-evaluation posts went to 14B consideration: `v2pjjn` "Cash Flow Analysis?", `1ntjzjx` "A 2018 Bolt. Feels too good to be true", `qzw5l6` "Rivian R1S or R1T | Is it worth ~$80k".
5. **Families: 14B over-uses `purchase_selection`** (17 posts vs Ryan's 4; only 3 overlap) — any post naming a model gets it, e.g. `qru24x`, `8n6b6g`, `4hnrf3`, `a32yh6`, `5vc0ev`. **Ryan marks `cost_incentives`, `policy_market`, `reliability_service` more** (11/10/15 vs 4/4/8), e.g. `nb5x7x` Doge down payment, `sqz3f4` dealer-markup mod thread, `1obmzbo` and `14uirtn` (policy_market for market-strategy chatter), `wp6ssp` and `mm5678` (reliability_service for owner experience).

## Is it good enough to report?

**No, not yet.** Stage κ is 0.16 (upper bound 0.30) and family κ is 0.44 (upper bound 0.56), both well under the ~0.6 "substantial" bar. The small model is worse on stage (κ 0.08). Only `charging_energy` (κ 0.81) clears it. Aspects are moderate (κ 0.50–0.56).

Limits on this read: n = 50, the calibration set only, a single human coder with no Daisy sheet, and the stage draw was stratified, so it is not a random sample of the corpus. Even so, the stage gap is large and systematic (awareness/unclear, LTO/onboarding), so it is unlikely to disappear with more rows.

## Suggested next steps

1. Daisy codes the same 50 calibration rows; compare with Ryan; freeze any codebook wording (notably: news/reviews/product reveals = `unclear`; owner experience with no "just bought" cue = `long_term_ownership`).
2. Consider a prompt revision for the 14B (explicit "news or a product reveal with no writer journey is unclear, not awareness"; "purchase_selection only if the writer is choosing") and re-score on these 50 before relabeling anything.
3. Ryan then codes the other 750; rescore with `score_agreement.py` as the codebook describes.

## Files

- `validation.csv` — data checks above
- `agreement_scores.csv` — all κ / α / % agreement with intervals, overall and by audience, per family and per aspect
- `confusion_stage_ryan_vs_qwen14b.csv`, `confusion_stage_ryan_vs_small.csv`
- `family_counts.csv` — positives per family for Ryan, 14B, and both
- `row_level_ryan_vs_models.csv` — every coded post with both sets of labels and Ryan's note
- `score_ryan.py` — reproduces everything here
