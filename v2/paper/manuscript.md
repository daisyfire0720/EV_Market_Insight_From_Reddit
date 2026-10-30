# Stage and topic in electric-vehicle talk on Reddit, 2015–2026

**Draft for Transportation Research Part D. The labels are unvalidated. No human has coded the annotation sheet, so there is no human kappa. Section 7 is the small model. Section 8 is Qwen2.5-14B. Neither is a finding about buyers. Read section 9 before the tables. A jump at a cutoff is consistent with a change in the posts. It is not an effect of the statute or the price cut.**

## Title options

1. Where the barrier sits in the purchase: stage and topic in electric-vehicle talk on Reddit, 2015–2026.
2. Enthusiast and shopper submissions around four US electric-vehicle shocks: a six-stage funnel.
3. Document-level stage, topic family, and barrier labels for electric-vehicle discussion on Reddit.

Working title used below: option 1.

## Abstract

Surveys of people who say they might buy an electric vehicle record price, range, and charging as barriers. They miss the month a credit or a price cut arrives, and they miss the wording used when nobody is asking. This paper reads Reddit submissions from 2015 through September 2026. Each submission gets one stage in a six-stage purchase funnel, one or more of eight topic families, a finer barrier aspect when one is present, and a stance. The stage is the writer's place in the journey, not a keyword. Enthusiast posts are a week sample of seven electric-vehicle subreddits. Shopper posts are keyword-matched submissions from r/cars, r/Autos, and r/carbuying. The labels in section 8 are from Qwen2.5-14B-Instruct (`qwen2.5:14b-instruct`), run locally in a 4-bit quant. `qwen2.5:7b-instruct` was the fallback and was not used. The smaller zero-shot model is reported beside the 14B labels. It is not the main labeler. Two annotators code 800 posts, including a 50-post calibration subset that is left out of kappa. Coder B is Ryan. That sheet is still blank, so this draft has no validated result.

The small-model file has 9044 enthusiast posts and 13783 shopper posts. That model has one ownership class, so section 7 cannot separate onboarding from long-term ownership. Section 8 reports the 14B labels on the same posts, with the small model beside them. Two jumps are consistent with the small model and neither interval excludes zero. Neither survives dropping posts with score under 2. Three other jumps fail a fake-date check and are not treated as results. A balanced comment sample with the parent title in the prompt does not reproduce those two jumps. Submissions stay the main specification.

## 1. Introduction and research questions

Electric-vehicle policy treats price, charging, and information as what keeps households out. Those claims are usually tested with surveys of people who already say they might buy. A survey misses the language used when the writer was not recruited, and it misses the particular day a credit rule or a price cut arrives.

Reddit is a public record of that language. Ruan and Lv (2022) recovered topics and a sentiment trend from 3,437,917 Reddit posts, mostly comments, from January 2011 through December 2020. Ruan and Lv (2023) showed that the Reddit pattern over that decade is not the Twitter pattern. Ashby, Weir, and Fussey (2025) organized charging talk on just under 100,000 tweets. Yu et al. (2025) showed what a document-level check looks like, on charging-station reviews rather than on Reddit. None of those papers measures, at the post and with a double-coded check, where the writer is in the purchase and what the post is about, through the dates that now organize US electric-vehicle buying.

The paper is built on a funnel. A post can be about charging at two different moments: while the writer is still deciding, and in the first week of ownership. Collapsing those into one topic, or into one ownership stage, hides the policy question. The taxonomy in `v2/docs/reference_taxonomy_6stage_8family.pdf` keeps six stages, with onboarding separate from long-term ownership, and eight topic families. The seven barriers already used in this project stay as the finer aspects.

The communities are part of the question, and a subreddit is not a person. Posts in r/electricvehicles, r/teslamotors, r/BoltEV, r/Rivian, r/Ioniq5, r/KiaEV6, and r/evcharging are the enthusiast file. Posts in r/cars, r/Autos, and r/carbuying that mention an electric vehicle are the shopper file. Owners post in r/cars, and shoppers post in r/teslamotors. The stage label, not the subreddit name, is what places a post in the funnel.

Four research questions:

1. How do topic-family and barrier shares differ across the six stages, and do those stage-by-family patterns differ between the enthusiast file and the shopper file?
2. At each of four cutoffs, does the share of posts about the pre-specified barrier jump: the Inflation Reduction Act on 16 August 2022 (incentives), the credit rules for vehicles placed in service after 31 December 2022 (incentives, cutoff 1 January 2023), Tesla's US price cuts on 13 January 2023 (price), and the end of the federal clean-vehicle credit on 30 September 2025 (incentives)?
3. Among authors who post more than once, how often does the next post sit further down the funnel, and is that rate higher for pairs that cross one of the four cutoffs?
4. After two people code the 800-post sheet, do the local instruction model and the small baseline model agree with the agreed human labels well enough to support the first three questions?

Volume is reported next to the shares. A change in how many posts were made is not a change in what the posts were about. A jump at a cutoff is not, by itself, an effect of the statute or the price cut. Hausman and Rapson (2018) are the reason to say so: a regression discontinuity in time is closer to an event study than to a cross-sectional discontinuity, because other news can land on the same day and because the bandwidth does not shrink in the way a cross-sectional design assumes.

The hypotheses that match those questions are in `v2/paper/introduction.md`. They are not tested in this draft.

## 2. Related work

### 2.1 Surveys of adoption barriers

The survey record used here is three papers whose publisher pages were opened, not a list this project had already named in its files. Egbue and Long (2012) [@egbue2012barriers] survey technology enthusiasts: battery range is the largest concern, then cost, and knowledge of the technology differs across groups. Rezvani, Jansson, and Bodin (2015) [@rezvani2015advances] review consumer-adoption studies: purchase cost is a barrier, and a negative evaluation of range remained after hands-on experience. Coffman, Bernstein, and Wee (2017) [@coffman2017revisited] review the same literature: public charging is associated with uptake, and much of the evidence is a survey about a hypothetical purchase. A paper that isolates an information barrier, as distinct from the knowledge difference in Egbue and Long, was not opened. This draft does not add one.

### 2.2 Social media as a record of electric-vehicle opinion

Ruan and Lv (2022) [@ruan2022reddit] fit a 20-topic LDA model and compared LIWC, VADER, and AFINN on Reddit through December 2020. The article is in *Communications in Transportation Research*, not Transportation Research Part D. Ruan and Lv (2023) [@ruan2023cross] repeat that corpus and add 7,383,327 tweets. Ashby, Weir, and Fussey (2025) [@ashby2025charging] use BERTopic and a language model on charging tweets, with manual review of the theme list rather than a double-coded test of document labels. The main text type in the present paper is the submission. Comments were labeled as a second text type and then left out of the discontinuity: the labeled batch is unbalanced around the cutoffs, and most comment stages are unclear. Ruan and Lv's mass is mostly comments, so a comparison of levels is still not like-for-like.

### 2.3 Language models as annotators

Gilardi, Alizadeh, and Kubli (2023) [@gilardi2023chatgpt] report that zero-shot ChatGPT beat crowd workers by about 25 percentage points on average across four samples (n = 6,183), and they treat validation as part of the method. Yu et al. (2025) [@yu2025equity] double-coded sentiment on 3,188 charging-station reviews (Cohen's κ = 0.81) and problem categories (human κ = 0.71); language-model accuracy on the categories was 0.77 to 0.82. Ha, Marchetto, Dharur, and Asensio (2021) [@ha2021topic] train a transformer on expert-annotated charging reviews. That paper is the template for a supervised classifier after a gold set exists. An accuracy figure for Ha et al. is not repeated here, because it was not re-checked against the publisher page in this project. VADER, the dictionary named by Ruan and Lv (2022), is Hutto and Gilbert (2014) [@hutto2014vader].

Cohen's kappa is Cohen (1960) [@cohen1960coefficient]. Krippendorff's alpha is the coefficient discussed by Krippendorff (2004) [@krippendorff2004reliability]. The prevalence correction computed by the scoring script is Rogan and Gladen (1978) [@rogan1978prevalence]. Those three entries were checked against the publisher record before they were added.

### 2.4 Discontinuity in time

Hausman and Rapson (2018) [@hausman2018rdit], *Annual Review of Resource Economics* 10:533–552, https://doi.org/10.1146/annurev-resource-121517-033306, set out how a regression discontinuity in time differs from a cross-sectional discontinuity: estimation often uses observations far from the cutoff, serial correlation can bias the jump, and a test for sorting around the cutoff is often beside the point. This paper uses that framing. The cross-sectional bandwidth papers used only as that contrast are Imbens and Kalyanaraman (2012) [@imbens2012bandwidth], on a data-driven bandwidth for the local-linear regression-discontinuity estimator, and Calonico, Cattaneo, and Titiunik (2014) [@calonico2014robust], on confidence intervals that depend on the bandwidth. This paper does not use their estimators.

## 3. Data

Counts of enthusiast submissions come from the Arctic Shift API (`https://arctic-shift.photon-reddit.com`). Pushshift no longer updates, and the official Reddit API needs credentials this project does not have. Complete months from January 2015 through September 2026 contain 603,435 submissions in the seven enthusiast subreddits. October 2026 was partial on 16 October 2026 (2,440 submissions).

The enthusiast text file is not that census. It is one deterministic 7-day window per subreddit-quarter, limit 25, plus a fixed week around each coded event, limit 40. The window rule is in `v2/download_reddit.py`. Round 6 added three further 7-day windows, limit 40, inside four baseline months that the week sample had left thin: August 2021, June 2022, December 2022, and September 2024. The cap on that fill was 3360 posts and 90 minutes of labeling; it appended 2642 posts in 17.55 minutes.

Shopper text is streamed from Academic Torrents zstd dumps (Watchful1, 2026, subreddit submissions 2005-06 to 2025-12, infohash `3e3f64dee22dc304cdd2546254ca1f8e8ae542b4`). The keyword filter is a word-boundary match for tesla, rivian, ioniq, "electric vehicle", and "electric car". Bare `ev` is excluded. The extract through December 2025 kept 13,976 posts. January–September 2026 kept 421 keyword posts, of which 413 survived cleaning. The API aggregate for the same three subreddits and the same keywords, January 2015 through September 2026, is 16,993. The two matchers are different, and the dump filter was not retuned to force the totals together. Comments are pulled separately, for the same subreddits and the windows around the four cutoffs, and they stay a second text type. The submission file is unchanged.

After cleaning, the labeled analysis file has 9044 enthusiast posts and 13783 shopper posts. A probable bot, or a post under five tokens, is dropped before labeling. Usernames are not in the committed label file. The annotation sheet redacts `u/` mentions. Full post text stays in gitignored `v2/data/raw/`.

The license field on the Academic Torrents item is empty. A takedown thread exists. The paper can report counts and the measurement method. It should not republish the dumps.

## 4. Measurement

### 4.1 The local instruction model, and the small model as baseline

The instrument for the corpus is Qwen2.5-14B-Instruct, served by Ollama as `qwen2.5:14b-instruct` in the Q4_K_M quant (about 9 GB of weights), with a context length of 4096 tokens. The finished run labeled the submissions and the 26,088 comments in batch 1 on one RTX 4070, about 13 hours. The planning figure of about 60 hours was the slow case used to decide whether the run would fit. It is not the time the run took. `qwen2.5:7b-instruct` and a rented GPU were fallbacks if the 100-post smoke test had been slower than about 7.4 minutes or the 14B model had run out of memory. They were not used. No paid API was used. The prompt is the codebook: one stage, one or more families, the finer aspects, a stance, and a short quote from the text. A comment is prefixed so the prompt says it is a comment, and the output row stores `text_type`. The main comment prompt does not include the parent post. The comments v2 sensitivity adds the parent title and a short parent excerpt. Section 8.7 reports that sample. It is not the main specification. The output is JSON. Each row stores the model name, the model version, and a hash of the prompt. The runner is `v2/llm_label.py`. Unknown family and aspect codes are dropped and the rest of the row is kept. A row whose stage is not in the codebook is left out of the tables, and the count is in the drop log (`row_drop_log.csv`), not a silent inner join. One duplicate submission was excluded before the merge. Section 8 names it.

The small model is the baseline reported beside the 14B labels. It is `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`, revision `262ae02f29173eec1c250f90804dc7edc677dcff`, on CPU. Barriers and stage use the template "This text is about {}.". Stance uses "This text expresses {}". A barrier is on when its score is at least 0.5. Dealer friction is also on when a small lexicon matches a markup, a documentation fee, a market adjustment, or a dealer inventory wait, and it is turned off when the model fires on ordinary service or on a charger that merely sits at a dealer and the lexicon did not match. That lexicon check is on invented sentences (`v2/fixtures/dealer_sentences.csv`). The small model has one ownership class. It does not emit the eight families. It is not the main labeler.

### 4.2 Taxonomy

The codebook is `v2/annotation/CODEBOOK.md`. Stage is one code: awareness, consideration, evaluation, purchase, onboarding, long-term ownership, or unclear. Onboarding is the first period of ownership (setup, the app, the first week). Long-term ownership is later use (degradation, warranty, service, resale, regret). Charging can be either, depending on whether the writer already has the car. Families are multi-label: purchase and vehicle selection; charging and energy access; battery, range, and real-world performance; cost, incentives, and financial feasibility; dealer, purchase process, and market access; reliability, service, and ownership experience; policy, regulation, and market environment; and ICE, hybrid, and EV transition narratives. Aspects are the seven barriers: charging access, driving range, price or total cost, purchase incentives or tax credits, battery reliability, dealer or buying friction, and environmental impact, plus an explicit none. Stance is support, oppose, mixed, or no stance. A complaint about one broken charger is not opposition by itself.

### 4.3 Annotation and validation

The sheet is `v2/annotation/annotation_sheet.csv`: 800 posts (382 enthusiast, 418 shopper), excerpts of at most 500 characters, code cells empty. The draw covers all six stages and all eight families in each audience, with rare cells filled first. A short lexicon was used only to place some of the small model's ownership posts into the onboarding pool and some into the long-term pool, so both stages appear on the sheet. That lexicon is not a code, and it is not printed on the sheet. Fifty rows are flagged `calibration=yes`. The two annotators code those first, compare them, and freeze the codebook. The 50 are excluded from kappa. The practice rows in `annotation_preview.csv` are also excluded. Model predictions are in `annotation_model_key.csv`, not on the sheet. Coder A is Daisy. Coder B is Ryan (GitHub Ryan-yu-wei). His instructions are `v2/annotation/README_FOR_RYAN.md`. He writes `Ryan` in the annotator column and saves `coder_ryan.csv`. This project did not contact him.

`v2/score_agreement.py` reads the two finished sheets. It writes Cohen's kappa and Krippendorff's alpha for each aspect, each family, stage, and stance, with a bootstrap interval for kappa. It scores a model only against items both coders marked and agreed on, and it leaves disagreements out of that gold set. It writes precision, recall, and F1, and a Rogan-Gladen correction of the prevalence. Blanks are missing, not a third category. The calibration rows are written to their own table and left out of the main one. The script was run on synthetic labels only. It does not invent a coder.

The rule already fixed is: if the lower bound of the confidence interval on the aspect kappas is below 0.60, add 200 adjudicated posts before a substantive claim, and if it is still too low, hold the Transportation Research Part D submission. The same comparison is then run for the small model and for the local instruction model. `v2/run_after_llm.py` does that second step when both coder files are passed in. Until those files exist, model-versus-human accuracy is not a number in this paper.

### 4.4 Score sensitivity

The main analysis keeps every item. A sensitivity drops items with score under 2. Missing scores are counted and left out of that check. They are not treated as a score of 2 or more. `v2/run_after_llm.py` writes `score_sensitivity.csv`, and, when scores are present, `funnel_shares_score_ge_2.csv`, `rdit_score_ge_2.csv` (point estimates only), and `stage_transition_shocks_score_ge_2.csv`. The main share, discontinuity, and transition tables stay on the full file.

`post_scores.csv` has 22826 submission rows: 12287 with score under 2, 10539 with score at least 2, and 0 with a blank score. 1 labeled submission id had no score row returned, so it is missing rather than a score of 0. That split is a property of the archive scores. It is not an instruction-model result. The 14B score check is in section 8. It uses this same rule. The main tables still keep every submission.

## 5. Discontinuity in time

### 5.1 Estimand

For a cutoff date c and a pre-specified aspect, let Y_i be 1 when post i is labeled with that aspect and 0 otherwise. Let X_i be the post's calendar day minus c, in days. The estimand is the jump in the conditional mean at the cutoff:

tau = limit as x approaches 0 from the right of E[Y | X = x] minus the limit as x approaches 0 from the left of E[Y | X = x].

Posts are aggregated to the day. Day t has n_t posts and share Y-bar_t. A local linear regression is fit separately on each side of the cutoff, with a triangular kernel K(x) = max(0, 1 - |x| / h), and with weight n_t K(x). tau-hat is the right-hand intercept minus the left-hand intercept. The outcome is incentives at the two credit cutoffs and price at the Tesla cutoff. The other aspects are not the pre-specified outcomes.

The 1 January 2023 cutoff is twelve days before 13 January 2023. The two fits are not separate experiments. Hausman and Rapson (2018) are also the reason not to read tau as the effect of the statute: another announcement can fall on the same day, and the enthusiast file is a week sample, so many days are empty.

### 5.2 Bandwidth, donut, and placebos

The bandwidth h is chosen from the grid 30, 60, 90, and 180 days by even/odd day cross-validation. Even days are used to fit the two boundary levels. Odd days are predicted from those levels, without a slope. The bandwidth with the lowest weighted squared error is the main one. The other three bandwidths are reported as a robustness grid and are not re-labeled as the main result.

The donut refits the chosen bandwidth after dropping posts with |X| at most 7 days. A thin fit has fewer than 30 posts or fewer than 5 distinct days on either side. A thin fit is reported and not interpreted.

Placebo cutoffs use the same outcome and the same chosen bandwidth, moved by 180 days earlier and 180 days later. A placebo that is as large as the main jump is a reason to doubt the main jump. It is not a second finding.

### 5.3 Inference

The interval is a percentile interval from a week-cluster bootstrap. Weeks inside the bandwidth are resampled with replacement, the daily shares are rebuilt, and the local linear jump is refit at the bandwidth chosen on the original sample. The bandwidth is not re-chosen inside each draw, so the interval is for that bandwidth. The main fit and the donut are resampled. The placebo and the bandwidth grid are point estimates. `v2/run_after_llm.py` writes the interval when it is pointed at a label file. The small-model table in section 7 is the point estimate only. An interval was not attached to that baseline table.

## 6. Stage transitions

Authors are hashed. Deleted accounts are dropped. The hash is not a name. Posts by the same hash are ordered by time. A pair is two successive posts. On the six-stage ladder the rungs are awareness, consideration, evaluation, purchase, onboarding, and long-term ownership, in that order. Unclear is not on the ladder. A pair is a forward move when both posts sit on the ladder and the later rung is higher. The rate is the share of pairs that move forward.

For a cutoff c, the shock rate uses pairs whose first post falls in the 180 days before c and whose second post falls in the 180 days after c. The same window is applied at c minus 180 days and at c plus 180 days. The comparison is descriptive. It is not a hazard model with covariates, and it is not an estimate of how fast a person moves from shopping to buying. The small-model baseline cannot see onboarding versus long-term ownership, so its ladder stops at one ownership rung. The 14B file can use the six-rung ladder, and section 8 does, on submissions only. Repeat-author coverage is the feasibility check. A week sample contains the same person twice less often than a census would. Comments are their own text type. Section 7.5 reports how many authors gain an item on the other side of a cutoff once comments are counted. That gain is coverage. It is not the discontinuity sample. Submissions are the main specification.

## 7. Preliminary baseline from the small model

**These numbers are the small zero-shot model. Section 8 puts the 14B labels beside them. Both are unvalidated. They are not a result about buyers.**

### 7.1 Stage shares

Ownership is one class on this model. Onboarding is not identified. The denominator is the labeled file for that audience.

| Stage on the small model | Enthusiasts | Shoppers |
|---|---:|---:|
| ownership | 0.387 (n=3504) | 0.629 (n=8670) |
| unclear | 0.314 (n=2839) | 0.043 (n=592) |
| consideration | 0.207 (n=1874) | 0.244 (n=3365) |
| purchase | 0.046 (n=418) | 0.057 (n=784) |
| awareness | 0.026 (n=231) | 0.024 (n=325) |
| evaluation | 0.020 (n=178) | 0.003 (n=47) |

### 7.2 Finer aspects

The aspect shares below are the small model's barrier indicators. The eight families are not in this file.

| Measure | Enthusiast share | Shopper share |
|---|---:|---:|
| charging access | 0.173 (n=9044) | 0.034 (n=13783) |
| driving range | 0.070 (n=9044) | 0.060 (n=13783) |
| vehicle price or total cost | 0.175 (n=9044) | 0.175 (n=13783) |
| purchase incentives or tax credits | 0.030 (n=9044) | 0.014 (n=13783) |
| battery reliability | 0.112 (n=9044) | 0.041 (n=13783) |
| dealer or buying friction | 0.033 (n=9044) | 0.027 (n=13783) |
| environmental impact | 0.066 (n=9044) | 0.047 (n=13783) |

### 7.3 Discontinuity, point estimates only

Local linear jump in the daily share. Bandwidth chosen on the grid in section 5. No week-cluster interval is attached to this baseline table. The Tesla enthusiast donut is unstable and is not interpreted.

| Cutoff | Audience | Outcome | Fit | Bandwidth (days) | Jump | Posts before | Posts after | Thin |
|---|---|---|---|---:|---:|---:|---:|---|
| ira_signing_2022 | enthusiast | purchase incentives or tax credits | main | 180 | 0.012 | 1228 | 1215 | False |
| ira_signing_2022 | enthusiast | purchase incentives or tax credits | donut | 180 | 0.030 | 1021 | 1199 | False |
| ira_credit_rules_2023 | enthusiast | purchase incentives or tax credits | main | 180 | 0.050 | 1231 | 1033 | False |
| ira_credit_rules_2023 | enthusiast | purchase incentives or tax credits | donut | 180 | 0.052 | 1215 | 1023 | False |
| tesla_price_cut_2023 | enthusiast | vehicle price or total cost | main | 60 | -0.078 | 923 | 148 | False |
| tesla_price_cut_2023 | enthusiast | vehicle price or total cost | donut | 60 | -0.393 | 762 | 104 | False |
| credit_end_2025 | enthusiast | purchase incentives or tax credits | main | 60 | 0.009 | 353 | 148 | False |
| credit_end_2025 | enthusiast | purchase incentives or tax credits | donut | 60 | 0.021 | 142 | 137 | False |
| ira_signing_2022 | shopper | purchase incentives or tax credits | main | 60 | -0.049 | 193 | 162 | False |
| ira_signing_2022 | shopper | purchase incentives or tax credits | donut | 60 | 0.004 | 170 | 141 | False |
| ira_credit_rules_2023 | shopper | purchase incentives or tax credits | main | 180 | 0.017 | 532 | 598 | False |
| ira_credit_rules_2023 | shopper | purchase incentives or tax credits | donut | 180 | 0.023 | 513 | 569 | False |
| tesla_price_cut_2023 | shopper | vehicle price or total cost | main | 90 | 0.131 | 277 | 297 | False |
| tesla_price_cut_2023 | shopper | vehicle price or total cost | donut | 90 | 0.051 | 246 | 256 | False |
| credit_end_2025 | shopper | purchase incentives or tax credits | main | 180 | -0.056 | 290 | 280 | False |
| credit_end_2025 | shopper | purchase incentives or tax credits | donut | 180 | -0.094 | 280 | 266 | False |

### 7.4 Whether authors post twice

| Audience | Posts with a hash | Authors | Authors with 2 or more posts | Share of posts from those authors | Median posts | Max posts |
|---|---:|---:|---:|---:|---:|---:|
| all | 19781 | 13903 | 2392 | 0.418 | 1.0 | 91 |
| enthusiast | 8489 | 6681 | 957 | 0.326 | 1.0 | 79 |
| shopper | 11292 | 7432 | 1404 | 0.466 | 1.0 | 91 |

### 7.5 Authors on both sides, with comments

Comments are a separate text type. An author is on both sides of a cutoff when at least one item falls in the 180 days before and at least one falls in the 180 days after. The submissions-only count is the robustness check. `n_gained` is the number of authors who meet that definition only after comments are included. The pull is `v2/fetch_comments.py`: comments by the hashed repeat authors, and comments on the sampled submissions, in the same subreddits and the same windows. Bodies stay in gitignored `v2/data/raw/`. The committed file is metadata.

| Cutoff | Submissions only | Submissions and comments | Authors gained |
|---|---:|---:|---:|
| ira_signing_2022 | 116 | 3723 | 3607 |
| ira_credit_rules_2023 | 115 | 3899 | 3784 |
| tesla_price_cut_2023 | 105 | 3810 | 3705 |
| credit_end_2025 | 39 | 2491 | 2452 |

The metadata file has 250794 comments from 34129 hashed authors. These counts are coverage, not labels and not effects.

Each side of a cutoff keeps the closest 100 comments returned for that author. If that page is full and none of them are in the study subreddits, the same window is requested again for the subreddits where that author submitted. A server timeout is retried on the same query.

Repeat authors searched: 2392 of 2392. Comments on the sampled submissions that fall inside the cutoff windows are included, one page of 100 comments per thread. The author total counts those commenters as well as the repeat submitters.

The run of record is 14B on the 22827 submissions plus `v2/outputs/comment_label_batch1.csv` (26088 comments, cutoff-crossing authors first). At the slow planning rate that pair is 60.0 hours. `v2/outputs/comment_label_batch2.csv` (224706 comments) stays optional and unlabeled. Neither batch file contains comment text. Bodies stay in the gitignored cache. These files are a queue, not labels. That queue has been labeled. Batch 1 is not usable for RDiT. The balanced comments v2 draw is a sensitivity in section 8. It does not replace the submission discontinuity. See `v2/COMMENTS_V2_PLAN.md`.

Pairs that cross a cutoff inside 180 days number about 100 on this file, and about 39 at the credit end. Forward rates on the unsplit ladder are about 0.13 to 0.18, and the placebos 180 days away fall in the same range. The median author has one post. A few hashes contribute many posts, so the repeat-post share is not a panel of distinct buyers. The counts are in `v2/outputs/stage_transition_shocks.csv`. Those pair counts are submissions. The comment gain above is a coverage count, not a new transition rate.

An earlier descriptive contrast, event-window share minus a baseline, with a design-aware bootstrap, is in `v2/outputs/FINDINGS.md`. It is not the estimand in section 5.

## 8. What the 14B labels show

**These are unvalidated model labels. Ryan has not coded the sheet, so the human kappa cell below is empty. A jump at a cutoff is consistent with a change in what was posted. It is not an effect of the statute or of the price cut. The main specification is submissions only.**

### 8.1 The labeler

The main labeler is Qwen2.5-14B-Instruct, served by Ollama as `qwen2.5:14b-instruct` in the Q4_K_M quant, temperature 0, prompt `funnel-v2`. The label file has 48,913 rows and 48,850 rows with a codebook stage. 63 rows put a topic-family code in the stage slot and are left out of the tables. The small zero-shot model in section 7 is the comparison, not the instrument. `qwen2.5:7b-instruct` was the fallback and was not the run. The finished labeling pass took about 13 hours on one RTX 4070. The planning rate of about 60 hours was the slow case, not the time that was used. Detail is in `v2/outputs/after_llm/FINDINGS_14B.md`.

### 8.2 Six-stage shares

The denominator is submissions with a codebook stage. Onboarding is separate from long-term ownership. The small model cannot make that split.

| Stage | Enthusiast share | Shopper share |
|---|---:|---:|
| awareness | 11.1% (n=9,036) | 32.7% (n=13,743) |
| consideration | 13.8% (n=9,036) | 14.6% (n=13,743) |
| evaluation | 10.7% (n=9,036) | 4.3% (n=13,743) |
| purchase | 2.8% (n=9,036) | 0.9% (n=13,743) |
| onboarding | 26.8% (n=9,036) | 4.3% (n=13,743) |
| long_term_ownership | 1.9% (n=9,036) | 2.0% (n=13,743) |
| unclear | 33.0% (n=9,036) | 41.1% (n=13,743) |

Enthusiast posts sit with people who already have the car: onboarding is 26.8%. Shopper posts sit earlier: awareness is 32.7%. Purchase and long-term ownership are thin in both files, under 3%. Unclear is about a third of enthusiast posts and about two fifths of shopper posts.

Dropping posts with score under 2 leaves the same picture: enthusiast onboarding 30.7%, shopper awareness 30.8%. That filter is a sensitivity. The table above keeps every submission.

### 8.3 Stage-by-family heatmap

Each cell is the share of submissions at that stage that carry the family. A post can carry more than one family. The grid is `v2/outputs/after_llm/stage_family_heatmap.csv` and the figure is `v2/outputs/after_llm/stage_family_heatmap.png`.

| Audience | Stage | Posts | Leading family | Share of that stage |
|---|---|---:|---|---:|
| enthusiast | awareness | 1,001 | purchase_selection | 66.4% |
| enthusiast | consideration | 1,243 | purchase_selection | 67.6% |
| enthusiast | evaluation | 969 | charging_energy | 56.3% |
| enthusiast | purchase | 253 | purchase_selection | 79.4% |
| enthusiast | onboarding | 2,421 | reliability_service | 50.8% |
| enthusiast | long_term_ownership | 171 | reliability_service | 78.4% |
| enthusiast | unclear | 2,978 | policy_market | 9.3% |
| shopper | awareness | 4,497 | purchase_selection | 80.5% |
| shopper | consideration | 2,005 | purchase_selection | 84.0% |
| shopper | evaluation | 592 | battery_range | 56.8% |
| shopper | purchase | 130 | purchase_selection | 80.8% |
| shopper | onboarding | 594 | reliability_service | 75.1% |
| shopper | long_term_ownership | 273 | reliability_service | 78.8% |
| shopper | unclear | 5,652 | policy_market | 14.8% |

The leading family is a description of the grid, not a test. Dealer friction and environment are the aspects the spot-read trusts least. See section 8.6.

### 8.4 Discontinuity, submissions only

The outcome is the pre-specified aspect: incentives at the credit cutoffs, price at the Tesla cutoff. The jump is in percentage points of the daily share. The interval is a week-cluster percentile interval from 1,000 draws, at the bandwidth chosen for the 14B labels. The small-model column is the same design on the section 7 labels, at the bandwidth that model chose. Bandwidths differ, so the grid rows in the two `rdit.csv` files are the fair comparison. None of these jumps is reported as an effect.

| Cutoff | Audience | 14B jump | 14B interval | Bandwidth | Placebo −180 / +180 days | Small model |
|---|---|---:|---|---:|---|---|
| ira_signing_2022 — demoted | enthusiast | +9.3 | [+2.3, +51.7] | 90d | +9.2 / -9.2 | +1.2 (180d) |
| ira_credit_rules_2023 | enthusiast | +3.5 | [-5.3, +7.2] | 180d | +1.3 / +5.4 | +5.0 (180d) |
| tesla_price_cut_2023 — demoted | enthusiast | -7.8 | [-17.6, -1.0] | 90d | -1.2 / -8.9 | -7.8 (60d) |
| credit_end_2025 | enthusiast | -0.5 | [-10.5, +6.4] | 180d | +3.3 / -1.5 | +0.9 (60d) |
| ira_signing_2022 | shopper | 0.0 | [-39.6, +26.9] | 30d | +7.4 / -14.9 | -4.9 (60d) |
| ira_credit_rules_2023 | shopper | +1.2 | [-10.4, +10.4] | 180d | -6.7 / +4.9 | +1.7 (180d) |
| tesla_price_cut_2023 | shopper | +10.5 | [-2.2, +24.2] | 180d | -6.2 / +4.9 | +13.1 (90d) |
| credit_end_2025 — demoted | shopper | -16.8 | [-32.3, +3.1] | 180d | -0.4 / +20.1 | -5.6 (180d) |

Two jumps are the ones to put beside the small model. Both intervals include zero. The language is "consistent with," not a claim that the rule or the price cut caused the posts.

Enthusiast posts about incentives around the 1 January 2023 credit rules: +3.5 points at 180 days, interval [-5.3, +7.2]. The small model, at its own bandwidth, is +5.0. On the 14B grid the jump is +5.4 at 60 days, and +4.3 at 90 days. That is consistent with the small-model direction. It is not distinguishable from no jump.

Shopper posts about price around the 13 January 2023 Tesla cuts: +10.5 points at 180 days, interval [-2.2, +24.2]. The small model, at its own bandwidth, is +13.1. On the 14B grid the jump is +15.3 at 60 days, and +11.9 at 90 days. That is consistent with the small-model direction. It is not distinguishable from no jump.

The upvote filter removes both of those jumps. The enthusiast credit-rules jump falls to +0.5 (n=786 before the cutoff and 667 after). The shopper Tesla price jump falls to +0.8 (n=187 before the cutoff and 171 after). The score check keeps 10,517 of 22,779 submissions and drops 12,262 with score under 2 (0 missing). Neither headline survives score at least 2. Report that as fragility. The main table still keeps every submission.

Three jumps fail the fake-date check in section 5. A placebo as large as the real date is a reason to doubt the jump. These three are not results.

- enthusiast, ira_signing_2022: +9.3 [+2.3, +51.7]. Placebos: +9.2 and -9.2. The placebo 180 days earlier is as large as the jump, and the days after the cutoff are thin.
- enthusiast, tesla_price_cut_2023: -7.8 [-17.6, -1.0]. Placebos: -1.2 and -8.9. The placebo 180 days later is as large as the jump.
- shopper, credit_end_2025: -16.8 [-32.3, +3.1]. Placebos: -0.4 and +20.1. The interval includes zero, and the placebo 180 days later is larger than the jump and points the other way.

### 8.5 Transitions, descriptive only

Forward moves on the six-rung ladder, for pairs that cross a cutoff inside 180 days: ira_signing_2022 20 of 116 (17.2%); ira_credit_rules_2023 12 of 115 (10.4%); tesla_price_cut_2023 15 of 105 (14.3%); credit_end_2025 4 of 39 (10.3%). Placebo rates run from 6.1% to 18.4%. No cutoff stands out from its fake dates. The counts are too thin for a claim. They describe the file. They do not estimate how fast a person moves from shopping to buying.

Author hashes on the submission file: 19,739 posts, 13,882 authors, 2,387 of them more than once (41.8% of those posts). The busiest hash has 91 posts.

### 8.6 Agreement

The first block is the small model against 14B. It is not a validity check. The human row is a placeholder. Ryan is coding `v2/annotation/annotation_sheet.csv`. This project did not contact him, and it does not invent his codes.

| Comparison | n | Agreement | Cohen's kappa |
|---|---:|---:|---:|
| Small model vs 14B, stage_comparable | 10,631 | 30.5% | 0.11 |
| Small model vs 14B, incentives_policy | 22,779 | 94.1% | 0.34 |
| Small model vs 14B, price_tco | 22,779 | 83.4% | 0.44 |
| Small model vs 14B, charging_access | 22,779 | 94.3% | 0.69 |
| Small model vs 14B, range | 22,779 | 92.4% | 0.41 |
| Small model vs 14B, battery_reliability | 22,779 | 86.1% | 0.20 |
| Small model vs 14B, dealer_friction | 22,779 | 88.9% | 0.17 |
| Small model vs 14B, environment | 22,779 | 84.5% | 0.06 |
| Small model vs 14B, family | 0 |  |  |
| Human kappa, Daisy and Ryan | — | — | placeholder. Ryan is coding. |

Stage kappa is about 0.11 on the posts where both models give a comparable stage. The small model has one ownership class, and those posts are not scored against onboarding or long-term ownership. Charging, price, and range are the aspects the two models share. Dealer friction and environment are low. A spot-read of 15 rows (seed 20261025, ids in `qc_spot_read_ids.csv`) found the model over-assigning aspects, especially those two. Treat them as low-confidence.

### 8.7 Comments are a sensitivity

Submissions are the main specification. Comment batch 1 is not usable for RDiT. The comments v2 draw is a sensitivity. It does not replace section 8.4, and it does not reproduce the two submission jumps.

#### Batch 1, no parent post

Batch 1 has 26,088 comment rows in the main label file. 26,071 have a codebook stage and an audience. Audience comes from the parent submission when that submission is in the study file (10,714 comments) and otherwise from the subreddit (15,357). Rows with neither are counted (`no_audience` = 0), not dropped in silence. `rdit_usable` is False. Batch-1 comments were selected by closeness to a cutoff, so the two sides are not a balanced window. This sensitivity is not usable for RDiT unless the sample is rebalanced. The main specification remains submissions only.

Without the parent post, comment stage is mostly unclear: enthusiast 64.6% (n=19,095); shopper 84.8% (n=6,976). The share table is `v2/outputs/after_llm/funnel_shares_comments.csv`.

Inside 180 days of the Tesla cutoff, enthusiast batch-1 comments with a codebook stage are 12,022 before the day, 1,495 on the day, and 1,470 after. Batch 1 picked comments by closeness to a cutoff. That design is not usable for RDiT.

#### Comments v2, parent title, balanced draw

Comments v2 follows `v2/COMMENTS_V2_PLAN.md`. `v2/outputs/comments_v2_draw_counts.csv` draws 8,000 comments, 500 on each side of each cutoff for each audience. Each comment is assigned to one cutoff, the nearest one, inside 180 days, and the cutoff day itself is left out. The prompt is `funnel-v2-comment-parent`: the comment, the parent title, and a short parent excerpt. The model is still `qwen2.5:14b-instruct`. The codebook is unchanged. Labels are in `v2/outputs/llm_labels_comments_v2.csv`. Tables are in `v2/outputs/after_llm_comments_v2/`. 7 comments returned no parseable JSON and are listed in `v2/outputs/comments_v2_parse_failures.csv`. They are left out, not imputed.

The parent title is worth keeping, and the lower unclear rate is a methods point, not a finding about buyers. A comment with no parent post often has no journey to code, which is why batch 1 is mostly unclear. With the parent title in the prompt, the unclear share is enthusiast 64.6% without the parent, 33.1% with it; shopper 84.8% without the parent, 67.3% with it. The two samples are not the same comparison. Batch 1 is the comments closest to a cutoff. Comments v2 is a random draw inside 180 days. The drop is therefore not an estimate of the prompt alone. It is still large enough, especially for enthusiasts, that a comment stage without the parent is the wrong instrument. Shopper comments stay mostly unclear even with the parent title, so the prompt does not turn every reply into a stage.

The comment jumps below are consistent with a flat comment share. They are not consistent with the submission point estimates, and they are not effects.

Shopper price talk at the 13 January 2023 Tesla cuts: +10.5 [-2.2, +24.2] in submissions, and -2.9 [-6.8, +2.7] on the comments drawn for that cutoff (+0.5 [-5.8, +7.0] if comments from the other cutoffs are pooled into the same window). The own-draw fit uses 5 days before the cutoff and 139 days after. The comment file does not reproduce the submission jump.

Enthusiast incentive talk at the 1 January 2023 credit rules: +3.5 [-5.3, +7.2] in submissions, and -5.6 [-8.7, +12.7] on the comments drawn for that cutoff. The own-draw fit uses 30 days before the cutoff and 6 days after. The comment file does not reproduce the submission jump.

The 1 January and 13 January cutoffs are twelve days apart. Assigning each comment to the nearest cutoff leaves about six days on the inner side: the days after 1 January, and the days before 13 January. The donut drops seven days around the cutoff, so that short side disappears and the donut fit is empty. That is 4 own-draw donut rows, one for each audience at each of those two dates (`v2/outputs/after_llm_comments_v2/rdit_comments_own_draw.csv`).

One of those short-side fits has an interval that excludes zero: shopper comments about purchase incentives or tax credits on 2023-01-01 are -4.9 [-10.5, -3.1], and the shorter side is 6 days. Its donut is empty. That interval is not a result, and it is not a submission headline.

The own-draw table is the one that matches the sample. Pooling comments across cutoffs mixes draws that were built for a different date. Neither reading supports the submission jumps. Submissions stay the main specification.

### 8.8 What was left out of the label file

One duplicate submission was excluded. `v2/outputs/EXCLUDED_IDS.txt` says: 13rktx5: model returned non-codebook JSON 3 times (deterministic decoding); exact duplicate title/text of 13rkt8q, which labeled fine. Excluded from merge on 2026-10-25.

Unknown codes outside the codebook: 761 of 48,913 rows (1.6%). The largest is `aspect:performance` (252 rows). An unknown family or aspect code is dropped and the rest of that row is kept. An unknown stage is left out of every table (63 rows: a family code where a stage should be). The raw code stays in `unknown_codes` when the label file has that column. A remap that moves level mixups back onto the right list, and that sets those 63 stages to unclear, changes funnel shares by at most 0.2 points and the main jumps by at most 0.3 points (`v2/outputs/after_llm_remapped/`). `performance` is not remapped. It is a codebook question, not a level mixup. See `v2/COMMENTS_V2_PLAN.md`.

## 9. Limitations

Read these before the tables in section 8.

The labels are unvalidated. Ryan is coding the 800-post sheet and has not returned it. There is no human kappa. The kappa table in section 8 leaves that cell as a placeholder. Until it is filled, the 14B shares and jumps are what a model said.

Stage agreement between the small model and 14B is about 0.11. Dealer friction and environment are the aspects the spot-read trusts least. Text was cut at 1,500 characters. Batch-1 comments were labeled without the parent post, and 65 to 85 percent of them are unclear. That batch is not usable for RDiT.

Comments v2 adds the parent title and draws 500 comments on each side of each cutoff. The unclear share is lower, and that is a reason to keep the parent in the prompt. The samples differ, so the drop is not a clean estimate of the prompt. The comment jumps do not reproduce the two submission headlines. The inner sides of 1 January 2023 and 13 January 2023 are about six days, and those donut fits are empty. Submissions stay the main specification.

One duplicate submission, `13rktx5`, was excluded. It is the same text as `13rkt8q`, which labeled normally. Unknown codes are 1.6 percent of rows. Unknown family and aspect codes are dropped and the rest of the row is kept. Sixty-three rows have a family code in the stage slot and are out of the tables. A remap that repairs the level mixups does not move the headlines.

Neither headline jump survives dropping posts with score under 2. The enthusiast incentive jump around 1 January 2023 and the shopper price jump around the Tesla cuts are consistent with the small model, and both intervals include zero. Three other jumps have a fake date as large as the real date and are not results: enthusiast incentives at the IRA signing, enthusiast price at the Tesla cuts, and shopper incentives at the credit end.

English-language Reddit is not the US household population. The enthusiast file is a cluster sample of submissions, not the 603,435-post census. The shopper file is keyword-filtered. Comments cover repeat authors and sampled threads in the cutoff windows, not every comment in the census. Arctic Shift does not guarantee that the archive is complete. The Academic Torrents license field is empty. Other news happens on the same days as the four cutoffs, so a jump is not an effect. The 1 January 2023 and 13 January 2023 cutoffs are twelve days apart. The small model cannot split ownership. The 14B model can, and that split is still a model label.

## 10. What remains before submission

1. Ryan (coder B) codes `v2/annotation/annotation_sheet.csv` against `v2/annotation/CODEBOOK.md`, calibration rows first, and returns `coder_ryan.csv`. The steps are `v2/annotation/README_FOR_RYAN.md`. This project did not contact him. His kappa replaces the placeholder in section 8.
2. The 14B run is finished. Do not rerun it to fill section 8. `qwen2.5:7b-instruct` stays the fallback that was not used. The smoke-test bound of about 7.4 minutes per 100 posts was the check that would have made a 60-hour plan unrealistic. The run that finished took about 13 hours.
3. Run `v2/score_agreement.py` on the two coder files. If the lower bound on aspect kappa is below 0.60, code 200 more posts before any substantive sentence.
4. Keep section 7 labeled as the small-model baseline. Keep the score-at-least-2 tables as the sensitivity. Do not describe a jump as an effect.
5. Comments v2 has been run. It is a sensitivity, not a new headline. Batch 2 stays unlabeled. Submissions stay the main specification. The checklist is `v2/paper/STATUS.md`.
6. Sources that were opened are in `v2/paper/references.bib`. Each one is marked in this draft with its BibTeX key. Do not add a DOI that has not been opened. A claim with no opened source is listed in `v2/paper/STATUS.md`.
7. Write the data-availability and ethics paragraphs the journal asks for: public posts, no attempt to identify users, dumps not republished, license field empty. Check the journal's current author guide for length and reference style. This draft does not guess those limits.

## References

Ashby, C., Weir, D., and Fussey, P. (2025) [@ashby2025charging]. Understanding public views on electric vehicle charging: A thematic analysis. *Transportation Research Interdisciplinary Perspectives*, 29, 101325. https://doi.org/10.1016/j.trip.2025.101325

Calonico, S., Cattaneo, M. D., and Titiunik, R. (2014) [@calonico2014robust]. Robust nonparametric confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295–2326. https://doi.org/10.3982/ECTA11757

Coffman, M., Bernstein, P., and Wee, S. (2017) [@coffman2017revisited]. Electric vehicles revisited: a review of factors that affect adoption. *Transport Reviews*, 37(1), 79–93. https://doi.org/10.1080/01441647.2016.1217282

Cohen, J. (1960) [@cohen1960coefficient]. A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46. https://doi.org/10.1177/001316446002000104

Egbue, O., and Long, S. (2012) [@egbue2012barriers]. Barriers to widespread adoption of electric vehicles: An analysis of consumer attitudes and perceptions. *Energy Policy*, 48, 717–729. https://doi.org/10.1016/j.enpol.2012.06.009

Gilardi, F., Alizadeh, M., and Kubli, M. (2023) [@gilardi2023chatgpt]. ChatGPT outperforms crowd workers for text-annotation tasks. *PNAS*, 120(30), e2305016120. https://doi.org/10.1073/pnas.2305016120

Ha, S., Marchetto, D. J., Dharur, S., and Asensio, O. I. (2021) [@ha2021topic]. Topic classification of electric vehicle consumer experiences with transformer-based deep learning. *Patterns*, 2, February 2021. https://doi.org/10.1016/j.patter.2020.100195

Hausman, C., and Rapson, D. S. (2018) [@hausman2018rdit]. Regression discontinuity in time: Considerations for empirical applications. *Annual Review of Resource Economics*, 10, 533–552. https://doi.org/10.1146/annurev-resource-121517-033306

Hutto, C., and Gilbert, E. (2014) [@hutto2014vader]. VADER: A parsimonious rule-based model for sentiment analysis of social media text. *Proceedings of the International AAAI Conference on Web and Social Media*, 8(1), 216–225. https://doi.org/10.1609/icwsm.v8i1.14550

Imbens, G., and Kalyanaraman, K. (2012) [@imbens2012bandwidth]. Optimal bandwidth choice for the regression discontinuity estimator. *The Review of Economic Studies*, 79(3), 933–959. https://doi.org/10.1093/restud/rdr043

Krippendorff, K. (2004) [@krippendorff2004reliability]. Reliability in content analysis: Some common misconceptions and recommendations. *Human Communication Research*, 30(3), 411–433. https://doi.org/10.1111/j.1468-2958.2004.tb00738.x

Rezvani, Z., Jansson, J., and Bodin, J. (2015) [@rezvani2015advances]. Advances in consumer electric vehicle adoption research: A review and research agenda. *Transportation Research Part D*, 34, 122–136. https://doi.org/10.1016/j.trd.2014.10.010

Rogan, W. J., and Gladen, B. (1978) [@rogan1978prevalence]. Estimating prevalence from the results of a screening test. *American Journal of Epidemiology*, 107(1), 71–76. https://doi.org/10.1093/oxfordjournals.aje.a112510

Ruan, T., and Lv, Q. (2022) [@ruan2022reddit]. Public perception of electric vehicles on reddit over the past decade. *Communications in Transportation Research*, 2, 100070. https://doi.org/10.1016/j.commtr.2022.100070

Ruan, T., and Lv, Q. (2023) [@ruan2023cross]. Public perception of electric vehicles on Reddit and Twitter: A cross-platform analysis. *Transportation Research Interdisciplinary Perspectives*, 21, 100872. https://doi.org/10.1016/j.trip.2023.100872

Yu, Q., Que, T., Cushing, L. J., Pierce, G., Shen, K., Kejriwal, M., Yao, Y., and Zhu, Y. (2025) [@yu2025equity]. Equity and reliability of public electric vehicle charging stations in the United States. *Nature Communications*, 16. https://doi.org/10.1038/s41467-025-60091-y

Watchful1. (2026) [@watchful1_2026]. Subreddit comments/submissions 2005-06 to 2025-12 [Data set]. Academic Torrents. Infohash `3e3f64dee22dc304cdd2546254ca1f8e8ae542b4`. The license field on the torrent is empty. No DOI.

Event dates, as used in `v2/research_design.md`: Pub. L. 117-169; 26 U.S.C. § 30D; Reuters, 13 January 2023, https://www.reuters.com/business/autos-transportation/tesla-cuts-prices-electric-vehicles-us-market-2023-01-13/; Reuters, 25 May 2023, https://www.reuters.com/business/autos-transportation/tesla-ford-ceos-talk-evs-twitter-forum-2023-05-25/; General Motors news release, 8 June 2023, https://investor.gm.com/news-releases/news-release-details/general-motors-doubles-down-commitment-unified-charging-standard/; Pub. L. 119-21; IRS, https://www.irs.gov/clean-vehicle-tax-credits.

