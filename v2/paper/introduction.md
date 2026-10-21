# Where the barrier sits in the purchase

Working title. Transportation Research Part D shape. This is the introduction and the research questions for the locked paper. The full draft is `manuscript.md` and `manuscript.tex`. Numbers below are counts of posts, not findings. Barrier shares from the small model are not evidence for the hypotheses.

## Introduction

Electric-vehicle policy treats price, charging, range, and information as the reasons households hold back. Those claims are usually tested with surveys of people who already say they might buy. A survey misses the wording people use when nobody has asked them, and it misses the particular day a credit or a price cut arrives.

Reddit is a public record of that wording. Ruan and Lv (2022) showed that the record is large enough to recover topics and a sentiment trend: 3,437,917 posts from January 2011 through December 2020, a 20-topic LDA model, and VADER. Ruan and Lv (2023) showed that the Reddit pattern is not the same pattern as Twitter over that same decade. Ashby, Weir, and Fussey (2025) showed that a topic model plus a language model can organize charging talk, on just under 100,000 tweets, with manual review of the themes. Yu, Que, Cushing, Pierce, Shen, Kejriwal, Yao, and Zhu (2025) showed what a check on document-level labels looks like, on charging-station reviews rather than on Reddit: two annotators, Cohen's κ of 0.81 for sentiment on 3,188 reviews, human inter-rater κ of 0.71 for problem categories, and language-model accuracy of 0.77 to 0.82 on those categories.

None of those papers measures, at the post and with that kind of check, where the writer is in the purchase and what the post is about, through the dates that now organize US electric-vehicle buying. The paper uses six stages, with onboarding kept separate from long-term ownership, and eight topic families. The note is `v2/docs/reference_taxonomy_6stage_8family.pdf`. A post about charging can be evaluation, if the writer is still deciding, or onboarding, if the car is already theirs. The stage is assigned from the writer's place in the journey, not from the keyword.

The dates for the discontinuity are the Inflation Reduction Act (Pub. L. 117-169, 16 August 2022), the credit rules for vehicles placed in service after 31 December 2022 (cutoff 1 January 2023; 26 U.S.C. § 30D as rewritten by that Act), Tesla's US price cuts reported on 13 January 2023, and the end of the federal clean-vehicle credit for vehicles acquired after 30 September 2025 (Pub. L. 119-21; 26 U.S.C. § 30D). The 1 January and 13 January cutoffs are twelve days apart. NACS remains in the descriptive record and is not one of the four discontinuity cutoffs. The Ruan corpora stop in December 2020.

The communities are part of the question. Posts in r/electricvehicles, r/teslamotors, r/BoltEV, r/Rivian, r/Ioniq5, r/KiaEV6, and r/evcharging are the enthusiast record. Posts in r/cars, r/Autos, and r/carbuying that mention an electric vehicle are the shopper record. A subreddit is not a person. Journey stage is a label on the post.

The instrument for the corpus is an open instruction model of about 7 to 8 billion parameters, run locally on an RTX 4070. The small zero-shot model already in `v2/outputs/framing_labels.csv` is the baseline. It has one ownership class and no topic-family field. Gilardi, Alizadeh, and Kubli (2023) are the reason to try a language model as an annotator, and the reason not to stop there. Two annotators, a frozen codebook, a 50-post calibration subset left out of kappa, and Cohen's kappa per aspect, family, stage, and stance are the check. Ha, Marchetto, Dharur, and Asensio (2021) are the template for a supervised classifier once those human labels exist. They are not a result in this draft.

The enthusiast census for complete months from January 2015 through September 2026 is 603,435 submissions. The labeled enthusiast file is a week sample, 9,044 posts after the thin-month fill. The shopper file is 13,783 keyword posts from the Academic Torrents dumps, including January–September 2026. The API keyword aggregate for the same three subreddits is 16,993. Those are counts of posts. They are not a finding about which barrier people raised.

## Research questions

1. How do topic-family and barrier shares differ across the six stages, and do those stage-by-family patterns differ between enthusiast submissions and shopper submissions?
2. At each of the four cutoffs above, does the share of posts about the pre-specified barrier (incentives, or price for the Tesla date) jump?
3. Among authors with more than one post, how often does the next post sit further down the funnel, and is that rate higher for pairs that cross one of the four cutoffs?
4. After the 800-post sheet is double-coded, do the local instruction model and the small baseline agree with the agreed human labels well enough to support the first three questions?

Submission volume is reported next to those shares. A change in how many posts were made is not a change in what the posts were about.

## Hypotheses

These are expectations to test after the human labels exist, on the instruction-model file, not on the small-model baseline. The pilot does not test them. A jump is the discontinuity in section 5 of `manuscript.md`. It is not an effect of the policy.

**H1. Incentives.** The incentive-aspect share jumps up at 16 August 2022, at 1 January 2023, and at 30 September 2025, relative to the local linear fit on the other side of the cutoff. The donut and the 180-day placebos are reported beside the main jump.

**H2. Price.** The price-aspect share jumps at 13 January 2023. That date is twelve days after 1 January 2023, so H1 and H2 are not separate experiments.

**H3. Stage by family.** Charging talk is not one cell. The charging family is more common in evaluation and in onboarding than in awareness. Price and dealer talk are more common at purchase than at awareness. The shopper file, once stage is labeled on the post, places more of its purchase-stage talk on price and on dealer friction than the enthusiast file does.

**H4. Transitions.** Among authors with at least two posts, the rate of forward moves is higher for pairs that cross one of the four cutoffs than for pairs that cross a placebo date 180 days away. This is a description of successive posts. It is not a claim that the shock caused a person to buy.

**H5. The instrument.** On the double-coded sample, with the 50 calibration posts left out, Cohen's kappa between the two annotators is high enough to support H1–H4. If the lower bound of the confidence interval on the aspect kappas is below 0.60, add 200 adjudicated posts before a substantive claim, and if it is still too low, hold the Transportation Research Part D claim. The small model is compared with the instruction model and with the agreed human labels. VADER is a stance baseline only. It is not a barrier classifier.

## What this draft does not claim

No human has coded the corpus. The instruction model has not been run. Nothing in the small-model baseline is a statement about what the public thinks. A jump at a cutoff is not an effect of the statute or of the price cut. English-language Reddit is not the US household population. The unit is the submission. Ruan and Lv's mass is mostly comments, so a comparison of levels with those papers is not like-for-like.

## Sources

Statute and news sources for the dates are in section 6 of `v2/research_design.md` and in the reference list of `manuscript.md`. Hausman and Rapson (2018), https://doi.org/10.1146/annurev-resource-121517-033306, is the discontinuity-in-time citation used in the methods. No other citation was added here.
