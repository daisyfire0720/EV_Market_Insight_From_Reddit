# Findings from the 14B instruction-model labels

Run date: 25 Oct 2026. Model: `qwen2.5:14b-instruct` (Q4_K_M) through Ollama, temperature 0, prompt `funnel-v2` (hash `a4ea383c018fc1b9`), on one RTX 4070.
Inputs: 22,825 submissions plus 26,088 batch-1 comments, 48,913 labels in `outputs/llm_labels.csv`.

**Nothing here is validated.** These are model labels and have not been checked against human codes. Every table carries `validated=False`. Treat each number as provisional until Ryan's second coding gives a kappa for the model against people.

## 1. What the tables cover

`run_after_llm.py` takes its audience labels from the small-model file (`framing_labels.csv`) and joins on them with an inner join. That file holds only submissions, so **every table in `outputs/after_llm/` is based on submissions only.** The 26,088 comments were labeled, but they never reach the funnel, RDiT, transition, or agreement tables.

- Of 22,825 submissions, 22,779 reach the tables. 46 were dropped because the model returned a topic-family code in the stage slot (see section 3).
- RDiT CIs are week-cluster bootstrap percentile intervals from 1,000 draws (`--n-boot 1000`). All three after_llm runs together took about 2 minutes.
- Comments are described in section 2. I also ran an exploratory version with comments included, kept on Daisy's PC only and not committed. Section 5 explains why it should not be used for RDiT.

## 2. Label quality checks (`qc_*.csv`, made by `qc_14b.py`)

### Stage share (%) by text type and audience

| stage | submissions, enthusiast (n=9,043) | submissions, shopper (n=13,782) | comments, enthusiast (n=19,104) | comments, shopper (n=6,984) |
|---|---|---|---|---|
| awareness | 11.1 | 32.6 | 3.5 | 4.9 |
| consideration | 13.7 | 14.5 | 7.8 | 5.6 |
| evaluation | 10.7 | 4.3 | 7.5 | 2.3 |
| purchase | 2.8 | 0.9 | 1.8 | 0.3 |
| onboarding | 26.8 | 4.3 | 13.3 | 1.2 |
| long_term_ownership | 1.9 | 2.0 | 1.4 | 0.8 |
| unclear | 32.9 | 41.0 | 64.6 | 84.8 |
| non-stage code | 0.1 | 0.3 | 0.1 | 0.1 |

Most comments are labeled `unclear` (65% enthusiast, 85% shopper). That is expected: comments are short and the model sees no parent post. Comments add little stage information unless the parent post is added to the prompt.

### Aspect share (%), multi-label

| aspect | sub, enth | sub, shop | com, enth | com, shop |
|---|---|---|---|---|
| price_tco | 11.4 | 22.9 | 9.2 | 7.7 |
| incentives_policy | 6.7 | 7.5 | 6.8 | 3.4 |
| charging_access | 22.7 | 3.9 | 16.3 | 2.4 |
| range | 7.9 | 7.0 | 5.5 | 3.1 |
| battery_reliability | 14.3 | 10.7 | 8.1 | 4.4 |
| dealer_friction | 14.1 | 9.3 | 8.2 | 5.1 |
| environment | 8.2 | 15.2 | 5.1 | 5.8 |
| no aspect | 35.8 | 48.8 | 55.9 | 77.9 |

### Topic-family share (%), multi-label

| family | sub, enth | sub, shop | com, enth | com, shop |
|---|---|---|---|---|
| purchase_selection | 21.1 | 43.8 | 10.2 | 9.6 |
| charging_energy | 22.6 | 3.7 | 16.2 | 2.3 |
| reliability_service | 19.2 | 12.4 | 8.7 | 4.8 |
| policy_market | 8.3 | 16.4 | 5.5 | 4.4 |
| battery_range | 7.9 | 5.5 | 5.3 | 2.9 |
| cost_incentives | 6.1 | 5.8 | 6.8 | 3.8 |
| transition_narratives | 2.7 | 6.9 | 3.2 | 3.7 |
| dealer_access | 3.4 | 1.7 | 2.1 | 0.9 |
| no family | 27.7 | 28.1 | 54.8 | 76.6 |

### Stance share (%)

| stance | sub, enth | sub, shop | com, enth | com, shop |
|---|---|---|---|---|
| support | 15.7 | 17.9 | 20.4 | 9.8 |
| oppose | 4.5 | 7.8 | 11.1 | 12.8 |
| mixed | 3.1 | 7.1 | 4.0 | 3.9 |
| no_stance | 76.6 | 67.1 | 64.5 | 73.5 |

### Codes outside the codebook (`qc_unknown_codes.csv`)

- **Rows flagged:** 761 of 48,913 (1.6%). That is 568 of 22,825 submissions (2.5%) and 193 of 26,088 comments (0.7%). There were 39 distinct invented codes.
- **Largest groups:**
  - `aspect:performance`: 252
  - `aspect:reliability_service`: 155
  - `family:environment`: 151
  - `aspect:dealer_access`: 46
  - `aspect:cost_incentives`: 27
  - `stage:dealer_access`: 26
  - `aspect:policy_market`: 21
  - `stage:policy_market`: 18
- **Pattern:** most cases are level mixups, where a family code is used as an aspect or the reverse. `performance` is the one real gap in the codebook.
- **How the main run handles them:** an unknown family or aspect code is dropped, and the rest of that row's labels are kept. A row with an unknown stage is left out of every table, because `usable_labels` keeps only codebook stages. The raw code stays in the `unknown_codes` column.

### The 63 rows with a non-stage code (`qc_unknown_stage_rows.csv`)

- **Where they are:** 46 submissions and 17 comments; 47 shopper and 16 enthusiast.
- **What the model returned:** in every case a topic-family code instead of a stage:
  - `dealer_access`: 26
  - `policy_market`: 18
  - `reliability_service`: 8
  - `transition_narratives`: 6
  - `cost_incentives`: 4
  - `purchase_selection`: 1
- **What the rows are:** almost all are news-style headlines with no buyer in them, for example "Dealers in Missouri sue to block Tesla" or "Legislation Would Let Tesla Sell in Texas". In practice the model meant "unclear stage, about this topic."

### Evidence quotes (`qc_evidence.csv`)

- **Empty evidence:** 37% of submissions and 56% of comments. The prompt allows an empty string.
- **Verbatim rate:** of the non-empty quotes, 95.3% of submission quotes and 93.0% of comment quotes appear verbatim in the text, ignoring case and whitespace. Ignoring punctuation as well, the rates are 96.1% and 94.8%.
- **The rest:** the remaining roughly 5% are paraphrases or quotes that were cut short.

### Spot-read of 15 random rows (seed 20261025; ids in `qc_spot_read_ids.csv`)

- **Plausible: 10 of 15.** These include:
  - the bill-text comment, labeled `policy_market`
  - the "$60k PHEV is high" comment, labeled consideration with price_tco
  - the dealer-supplied home charger comment, labeled onboarding with charging
  - several short or headline-only rows correctly labeled `unclear`
- **Partly off: 3.**
  - A hybrid-STI question was tagged `environment`.
  - A Bolt owner asking about Tesla-station cable reach was put at `evaluation`; it is closer to ownership.
  - An OEM-roadmap comment was tagged `dealer_friction`.
- **Clearly wrong: 1.** An autonomy comment got 3 families and 3 aspects (dealer, battery, environment), none of which fit.
- **Pattern:** the model's stages are mostly sensible, but it **over-assigns aspects**, especially `dealer_friction` and `environment`. Agreement with the small model points the same way: kappa is 0.17 for dealer and 0.06 for environment.

## 3. Remap sensitivity (`outputs/after_llm_remapped/`, log in `qc_remap_log.csv`)

**Rules:**

- An aspect slot holding a family code moves to families. Examples: `aspect:dealer_access` becomes family `dealer_access`, and `aspect:reliability_service`, `cost_incentives`, `policy_market` and `battery_range` move the same way.
- `aspect:reliability` and `aspect:maintenance` become family `reliability_service`.
- `aspect:policy` and `aspect:regulations` become family `policy_market`.
- `aspect:resale_value` becomes aspect `price_tco`.
- A family slot holding an aspect code moves to aspects, for example `family:environment` becomes aspect `environment`.
- A stage slot holding a family code puts that code in families and sets the stage to `unclear`. That brings all 63 rows back.
- `performance` and 15 other one-off codes stay dropped, 279 code instances in all.

**Result: no headline moves.**

- **Funnel shares:** change by at most 0.2 points.
- **RDiT:** every main estimate changes by at most 0.3 points, and the CIs are essentially identical.
- **Transitions:** the forward rates are identical.

## 4. Headline results (submissions only, 1,000 bootstrap draws)

### Funnel share by audience (`funnel_shares_by_audience.csv`)

| stage | enthusiast (n=9,036) | shopper (n=13,743) |
|---|---|---|
| awareness | 11.1% | 32.7% |
| consideration | 13.8% | 14.6% |
| evaluation | 10.7% | 4.3% |
| purchase | 2.8% | 0.9% |
| onboarding | 26.8% | 4.3% |
| long_term_ownership | 1.9% | 2.0% |
| unclear | 33.0% | 41.1% |

- **Enthusiasts:** posts are mostly from people who already own the car; onboarding (charger installs, delivery, first weeks) is the largest stage.
- **Shoppers:** posts sit at awareness and consideration.
- **Purchase and long-term ownership** are thin in both groups, under 3%.
- **Score >= 2 subset:** the picture is the same. Enthusiast onboarding is 30.7% and shopper awareness 30.8%.

### RDiT at the four cutoffs (jump in share of posts with the aspect, percentage points, 95% CI)

| cutoff | audience | outcome | 14B jump [95% CI] | bandwidth | donut 7d | placebo -180d / +180d | small model (same design) |
|---|---|---|---|---|---|---|---|
| IRA signing 2022-08-16 | enthusiast | incentives | +9.3 [+2.3, +51.7] | 90d | +7.6 [-9.2, +69.7] | +9.2 / -9.2 | +1.2 |
| IRA signing | shopper | incentives | 0.0 [-39.6, +26.9] | 30d | -9.0 | +7.4 / -14.9 | -4.9 |
| Credit rules 2023-01-01 | enthusiast | incentives | **+3.5 [-5.3, +7.2]** | 180d | +3.7 [-5.8, +7.2] | +1.3 / +5.4 | **+5.0** |
| Credit rules | shopper | incentives | +1.2 [-10.4, +10.4] | 180d | +1.0 | -6.7 / +4.9 | +1.7 |
| Tesla price cuts 2023-01-13 | enthusiast | price/TCO | -7.8 [-17.6, -1.0] | 90d | -4.8 [-28.3, +3.9] | -1.2 / -8.9 | -7.8 |
| Tesla price cuts | shopper | price/TCO | **+10.5 [-2.2, +24.2]** | 180d | +12.6 [-1.5, +23.2] | -6.2 / +4.9 | **+13.1** (90d) |
| Credit end 2025-09-30 | enthusiast | incentives | -0.5 [-10.5, +6.4] | 180d | +1.1 | +3.3 / -1.5 | +0.9 |
| Credit end | shopper | incentives | -16.8 [-32.3, +3.1] | 180d | -20.3 [-34.7, -0.7] | -0.4 / +20.1 | -5.6 |

**In plain terms:**

- **Enthusiast incentive talk at 1 Jan 2023:**
  - The 14B labels point the same way as the small model, but the jump is smaller: +3.5 points against +5.0.
  - At a 90-day bandwidth it is +4.3; at 60 days, +5.4.
  - The CI includes zero, so this is **consistent with but weaker than** the small-model result.
- **Shopper price talk after the Tesla cuts:**
  - +10.5 points at the chosen 180-day bandwidth, +11.9 at 90 days (the small model's bandwidth), and +15.3 at 60 days.
  - The small model gave +13.1.
  - The size is replicated. The CI, [-2.2, +24.2], just includes zero, and the donut version is +12.6.
  - The placebos are small and of either sign (-6.2 and +4.9), which is cleaner than the small model, whose -180-day placebo was +10.1.
- **Do not lean on these:**
  - The enthusiast IRA-signing jump (+9.3): the -180-day placebo is just as large, and post-cutoff data cover only 23 of 90 days.
  - The enthusiast Tesla drop (-7.8): the +180-day placebo is just as large.
  - The shopper credit-end drop (-16.8): its CI includes zero, and the +180-day placebo is +20.
- **Score >= 2 subset (no CI):** shopper Tesla price falls to +0.8 (n about 180 per side), and enthusiast credit rules to +0.5. Neither headline survives the upvote filter. This should be reported as a fragility.

### Stage transitions (same author, consecutive posts within 180 days of each cutoff)

- **Forward moves:**
  - IRA signing: 20 of 116 pairs (17%)
  - credit rules: 12 of 115 (10%)
  - Tesla cuts: 15 of 105 (14%)
  - credit end: 4 of 39 (10%)
- **Placebo rates** range from 6% to 18%, so no cutoff stands out.
- **Most pairs stay put:** unclear→unclear (1,412) and awareness→awareness (707) are the most common transition counts.
- **Score >= 2:** the rates are 16–19% on 18–81 pairs.
- Transition analysis is too thin with submissions alone to support any claim.

### Author coverage

- **Totals:** 19,739 submissions with an author hash, from 13,882 authors. 2,387 authors posted more than once, accounting for 41.8% of those posts.
- **By audience:** repeat authors account for 32.6% of enthusiast posts and 46.6% of shopper posts. The busiest single author has 91 posts.

### Small model against 14B (`small_vs_llm.csv`; this is not a validity check)

- **Stage:** 30.5% agreement, kappa 0.11, on the 10,631 posts where both models give a comparable stage. Another 12,148 posts carry the small model's single "ownership" label, which can't be split into onboarding or long-term ownership.
- **Aspect kappas:**
  - charging 0.69
  - price 0.44
  - range 0.41
  - incentives 0.34
  - battery 0.20
  - dealer 0.17
  - environment 0.06
- **What this means:** the two models broadly agree on charging, price and range. They disagree on stage and on the soft aspects. That is why human validation matters before the 14B labels replace the small model.

## 5. Comments: what to do with them

- **Comments are not in the tables:** the 26,088 comment labels exist but are left out of all the tables above, because of the audience join.
- **The exploratory run:** I joined them on subreddit audience in a separate run (`outputs/after_llm_with_comments`, not committed). Two problems showed up:
  1. **Unbalanced sample:** batch-1 comments were picked by how close they are to a cutoff, and the sides are very unbalanced. At the Tesla cutoff, enthusiast comments are 13,374 left against 2,290 right. Pooling them gives RDiT estimates that swing wildly with bandwidth and have intervals more than 100 points wide.
  2. **Mostly unclear:** 65–85% of comments are labeled `unclear`.
- **Recommendation:** keep comments out of the headline RDiT. Use them only in a separate, cutoff-specific design with balanced windows. If stage matters, relabel comments with the parent title in the prompt.

## 6. Caveats

1. **Unvalidated:** no human comparison yet. The headline numbers can't go into the paper as findings until Ryan's second coding gives a kappa for 14B against people.
2. **One duplicate excluded:** submission `13rktx5` returned non-codebook JSON 3 times under deterministic decoding. It is an exact duplicate of `13rkt8q`, which labeled normally. It is excluded (`EXCLUDED_IDS.txt`), leaving 22,825 of 22,826 submissions.
3. **63 rows with a family code as stage** are dropped in the main run and restored as `unclear` in the remap. The results don't change either way.
4. **Unknown family and aspect codes** (1.6% of rows) are dropped in the main run, and the raw codes are kept in `unknown_codes`. Remapping the level mixups doesn't change any result.
5. **Aspect over-assignment:** `dealer_friction` and `environment` are probably over-assigned (spot-read, and kappas against the small model of 0.17 and 0.06). Treat those two aspects as low-confidence.
6. **Text truncation:** text is cut at 1,500 characters, and comments are labeled without their parent post.
7. **Bandwidth choice:** the bandwidth is chosen by cross-validation separately for each model and audience. When comparing with the small model, use the bandwidth-grid rows in `rdit.csv`.
8. **Fragile to upvotes:** the score >= 2 subset erases both headline jumps.

## 7. Recommended manuscript updates

1. **Methods:** describe the 14B labeler, covering:
   - model, quantization, temperature 0 and prompt hash
   - the codebook-only instruction
   - how unknown codes are handled
   - the excluded duplicate
   - labeling 48,913 items in about 13 hours on one consumer GPU
2. **Funnel section:** the six-stage split replaces the small model's single "ownership". The headline is that enthusiast posts are onboarding-heavy (27%) and shopper posts awareness-heavy (33%).
3. **Discontinuity section:** report both labelers side by side.
   - The shopper price jump after the Tesla cuts replicates in size (+10 to +15 points depending on bandwidth), but the 95% interval includes zero.
   - The enthusiast incentive jump at 1 Jan 2023 is smaller (+3.5) and not distinguishable from zero.
   - Report the score >= 2 fragility.
   - Phrase all of this as "consistent with", not "effect of".
4. **Drop or demote** the enthusiast IRA-signing and Tesla results and the shopper credit-end result, because their placebos are as large as the main estimates.
5. **Transitions:** report them as descriptive only (10–17% forward at every cutoff, with placebos in the same range).
6. **Validity section:** add the small-model-vs-14B kappas, plus a placeholder for the human kappa.
7. **Limitations:** add that comments were labeled but left out, and why.
