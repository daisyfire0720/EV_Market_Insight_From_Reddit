# Purchase barriers in electric-vehicle talk on Reddit, 2015–2026

**Draft for Transportation Research Part D. All model results in this file are preliminary. No human has coded the annotation sheet. Do not cite the shares as findings.**

## Title options

1. Purchase barriers in public electric-vehicle discussion: Reddit submissions around the Inflation Reduction Act, Tesla's price cuts, NACS, and the 2025 credit deadline.
2. Enthusiast and shopper talk about electric-vehicle barriers on Reddit, 2015–2026.
3. Document-level barrier labels for electric-vehicle discussion on Reddit around four US market shocks.

Working title used below: option 1.

## Abstract

Surveys of electric-vehicle intenders record price, range, and charging as barriers, but they miss the months in which a credit, a price cut, or a charging-standard announcement lands, and they miss the wording people use when nobody is asking. This paper measures seven purchase barriers, purchase-journey stage, and stance on Reddit submissions from 2015 through September 2026. The enthusiast file is a week sample of seven electric-vehicle subreddits drawn from the Arctic Shift archive. The shopper file is keyword-matched submissions from r/cars, r/Autos, and r/carbuying, read from Academic Torrents dumps. A small open zero-shot model assigns the labels on CPU. Two annotators are planned for an 800-post sheet; that sheet is still blank, so every share below is unvalidated.

On the current unvalidated labels the enthusiast file has 9044 posts and the shopper file has 13783. Charging and battery are more common in the enthusiast file. Price is close in the two files. The shopper file is not a file of people who are about to buy: most of those posts are labeled as ownership or support, which is what the keyword filter returns. Around the four shocks, some event-minus-baseline differences exclude zero under a design-aware bootstrap and some do not. An interrupted series that resamples months agrees with fewer of them. None of those differences is an effect of the statute or the price cut.

## 1. Introduction and contribution

Electric-vehicle policy treats price, charging, and information as what keeps households out. Those claims are usually tested with surveys of people who say they might buy. A survey misses the language used when the writer was not recruited, and it misses the particular month a credit or a price cut arrives.

Reddit is a public record of that language. Ruan and Lv (2022) recovered topics and a sentiment trend from 3,437,917 Reddit posts (mostly comments) from January 2011 through December 2020. Ruan and Lv (2023) showed that the Reddit pattern over that decade is not the Twitter pattern. Ashby, Weir, and Fussey (2025) organized charging talk on just under 100,000 tweets. Yu et al. (2025) showed what a document-level check looks like, on charging-station reviews rather than on Reddit. None of those papers measures, at the post and with a double-coded check, the barrier a writer is raising, the stage of the purchase, and the stance, through the dates that now organize US electric-vehicle buying. Those dates are the Inflation Reduction Act (Pub. L. 117-169, 16 August 2022), Tesla's US price cuts reported on 13 January 2023, Ford's charging agreement of 25 May 2023 and GM's announcement of 8 June 2023, and the end of the federal clean-vehicle credit for vehicles acquired after 30 September 2025 (Pub. L. 119-21, 4 July 2025; 26 U.S.C. § 30D).

The question is how the barriers people discuss — price, charging, battery, range, incentives, dealer friction, and environment — sit around those four shocks, and whether enthusiast subreddits and shopper subreddits move together. Owner versus shopper is a contrast inside the paper, using the stage label on the post, not the name of the subreddit. Volume ratios are descriptive. They are not effects.

Three contributions, once the human labels exist:

1. A document-level codebook of adoption barriers, journey stage, and stance, with two-annotator agreement reported the way Yu et al. (2025) report it.
2. A comparison of that instrument with VADER and LDA, the tools in Ruan and Lv (2022), and with the keyword stage rule already in this repository.
3. A description of barrier shares around the 2022–2025 dates, in enthusiast communities and in r/cars, r/Autos, and r/carbuying, with uncertainty that follows the sample design.

The hypotheses are written in `v2/paper/introduction.md`. They are not tested in this draft.

## 2. Related work

### 2.1 Surveys of adoption barriers

Price, range, charging access, and information are the usual survey list. The survey papers Daisy already cites belong here. [CITATION NEEDED]

### 2.2 Social media as a record of electric-vehicle opinion

Ruan and Lv (2022) fit a 20-topic LDA model and compared LIWC, VADER, and AFINN on Reddit through December 2020. The article is in *Communications in Transportation Research*, not Transportation Research Part D. Ruan and Lv (2023) repeat that corpus and add 7,383,327 tweets. Ashby, Weir, and Fussey (2025) use BERTopic and a language model on charging tweets, with manual review of the theme list rather than a double-coded test of document labels. The unit in the present paper is the submission. Ruan and Lv's mass is mostly comments, so a comparison of levels is not like-for-like.

### 2.3 Language models as annotators

Gilardi, Alizadeh, and Kubli (2023) report that zero-shot ChatGPT beat crowd workers by about 25 percentage points on average across four samples (n = 6,183), and they treat validation as part of the method. Yu et al. (2025) double-coded sentiment on 3,188 charging-station reviews (Cohen's κ = 0.81) and problem categories (human κ = 0.71); language-model accuracy on the categories was 0.77 to 0.82. Ha, Marchetto, Dharur, and Asensio (2021) train a transformer on expert-annotated charging reviews. That paper is the template for a supervised classifier after a gold set exists. An accuracy figure for Ha et al. is not repeated here, because it was not re-checked against the publisher page in this project. VADER is named by Ruan and Lv (2022); the primary VADER paper was not opened here. [CITATION NEEDED]

Cohen's kappa, Krippendorff's alpha, and the Rogan-Gladen prevalence correction are the agreement statistics the scoring script computes. The bibliographic entries for those three methods were not opened in this project. [CITATION NEEDED]

## 3. Data

Counts of enthusiast submissions come from the Arctic Shift API (`https://arctic-shift.photon-reddit.com`). Pushshift no longer updates, and the official Reddit API needs credentials this project does not have. Complete months from January 2015 through September 2026 contain 603,435 submissions in r/electricvehicles, r/teslamotors, r/BoltEV, r/Rivian, r/Ioniq5, r/KiaEV6, and r/evcharging. October 2026 was partial on 16 October 2026 (2,440 submissions). Year totals are in `v2/outputs/FINDINGS.md`.

The enthusiast text file is not that census. It is one deterministic 7-day window per subreddit-quarter, limit 25, plus a fixed week around each coded event, limit 40. The window rule is in `v2/download_reddit.py`. Round 6 added three further 7-day windows, limit 40, inside four baseline months that the week sample had left thin: August 2021, June 2022, December 2022, and September 2024. The cap on that fill was 3360 posts and 90 minutes of labeling; it appended 2642 posts in 17.55 minutes.

Shopper text is not from the search API. It is streamed from Academic Torrents zstd dumps (Watchful1, 2026, subreddit submissions 2005-06 to 2025-12, infohash `3e3f64dee22dc304cdd2546254ca1f8e8ae542b4`). The keyword filter is a word-boundary match for tesla, rivian, ioniq, "electric vehicle", and "electric car". Bare `ev` is excluded. The extract through December 2025 kept 13,976 posts. January–September 2026 was streamed one monthly submissions file at a time from the Academic Torrents monthly dumps; 421 keyword posts were kept and 413 survived cleaning. The API aggregate for the same three subreddits and the same keywords, January 2015 through September 2026, is 16,993. The two matchers are different, and the dump filter was not retuned to force the totals together. Comments were not downloaded.

After cleaning, the labeled analysis file has 9044 enthusiast posts and 13783 shopper posts. A probable bot, or a post under five tokens, is dropped before labeling. Usernames are not in the committed label file. The annotation sheet redacts `u/` mentions. Full post text stays in gitignored `v2/data/raw/`.

The license field on the Academic Torrents item is empty. A takedown thread exists. The paper can report counts and the measurement method. It should not republish the dumps.

## 4. Methods

### 4.1 Labeling pipeline

The model is `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`, revision `262ae02f29173eec1c250f90804dc7edc677dcff`, on CPU. No paid API is used. Barriers and stage use the template "This text is about {}.". Stance uses "This text expresses {}". Those templates were chosen on invented codebook sentences, not by fitting the corpus. A barrier is on when its score is at least 0.5. Dealer friction is also on when a small lexicon matches a markup, a documentation fee, a market adjustment, or a dealer inventory wait, and it is turned off when the model fires on ordinary service or on a charger that merely sits at a dealer and the lexicon did not match. The lexicon comparison is on invented sentences (`v2/fixtures/dealer_sentences.csv`).

### 4.2 Taxonomy

The codebook is `v2/annotation/CODEBOOK.md`. Barriers are multi-label: charging access, driving range, vehicle price or total cost, purchase incentives or tax credits, battery reliability, dealer or buying friction, environmental impact, plus an explicit none. Stage is single-label: learning, comparing, evaluating cost or fit, buying, owning, or no clear stage. Stance is support, opposition, mixed, or no stance. Whether ownership is one code or is split into onboarding versus long-term ownership is an open decision. The sheet has an optional onboarding column so either choice can be scored without redrawing the sample.

### 4.3 Event windows

The event month is August 2022 for the Inflation Reduction Act, January 2023 for the Tesla price cuts, May and June 2023 together for NACS, and September and October 2025 for the credit deadline. July 2025, the month the statute was signed, is in the volume table and is not a fifth interruption. For each event the pre-specified barrier is incentives (IRA and the credit deadline), price (Tesla), or charging (NACS). The other barriers are reported beside it. Each contrast is the event-window share minus the prior three months, and minus the same months a year earlier.

### 4.4 Interrupted time series and the bootstrap

The interrupted series is a weighted least-squares level shift on monthly shares, twelve months before the first event month and six months after, dropping months with fewer than five posts. Calendar time is used, so a missing month is not renumbered into the next one. HC1 intervals treat months as independent. The design-aware interval resamples the months in that window and refits. A month drawn twice keeps both copies.

The event-window interval resamples clusters, not posts. Enthusiast clusters are the sampled week: the event week if the post falls in one, otherwise the quarterly window, otherwise a fill week from the round 6 pull, otherwise the subreddit-month. Shopper clusters are the calendar month when both sides have at least two months, and the subreddit-month otherwise. Two thousand resamples, seed 20261019. The point estimate does not change. A window built from fewer months than the design asks for is flagged.

## 5. Preliminary results

**These numbers are model output. They will move when human labels replace them, and some of them already moved when the thin baseline months were filled. They are not a result about buyers.**

### 5.1 Barrier shares

| Measure | Enthusiast share | Shopper share |
|---|---:|---:|
| charging access | 0.173 (n=9044) | 0.034 (n=13783) |
| driving range | 0.070 (n=9044) | 0.060 (n=13783) |
| vehicle price or total cost | 0.175 (n=9044) | 0.175 (n=13783) |
| purchase incentives or tax credits | 0.030 (n=9044) | 0.014 (n=13783) |
| battery reliability | 0.112 (n=9044) | 0.041 (n=13783) |
| dealer or buying friction | 0.033 (n=9044) | 0.027 (n=13783) |
| environmental impact | 0.066 (n=9044) | 0.047 (n=13783) |
| owning an electric vehicle | 0.387 (n=9044) | 0.629 (n=13783) |
| support for electric vehicles | 0.513 (n=9044) | 0.815 (n=13783) |

Figure: `v2/outputs/framing_barrier_shares_monthly.png`. Independent-post intervals on that figure are narrower than the week and month bootstrap in the tables below.

### 5.2 Event windows

The difference is the event share minus the baseline share. "Includes 0" refers to the design-aware bootstrap interval, not the independent-post interval.

| Event | Audience | Baseline | Difference | Bootstrap interval | Baseline posts | Includes 0 | Thin month |
|---|---|---|---:|---|---:|---|---|
| ira_2022 | enthusiast | prior_3_months | 0.042 | -0.001 to 0.087 | 897 | yes | no |
| ira_2022 | enthusiast | year_ago | 0.041 | -0.003 to 0.088 | 485 | yes | no |
| ira_2022 | shopper | prior_3_months | 0.001 | -0.041 to 0.147 | 277 | yes | no |
| ira_2022 | shopper | year_ago | 0.022 | 0.000 to 0.167 | 95 | yes | no |
| tesla_price_cut_2023 | enthusiast | prior_3_months | 0.073 | -0.026 to 0.162 | 845 | yes | no |
| tesla_price_cut_2023 | enthusiast | year_ago | 0.161 | 0.035 to 0.270 | 58 | no | no |
| tesla_price_cut_2023 | shopper | prior_3_months | 0.097 | 0.025 to 0.840 | 260 | no | no |
| tesla_price_cut_2023 | shopper | year_ago | 0.127 | -0.737 to 0.874 | 125 | yes | no |
| nacs_2023 | enthusiast | prior_3_months | -0.069 | -0.233 to 0.064 | 201 | yes | no |
| nacs_2023 | enthusiast | year_ago | 0.002 | -0.071 to 0.081 | 809 | yes | no |
| nacs_2023 | shopper | prior_3_months | 0.028 | -0.008 to 0.068 | 269 | yes | no |
| nacs_2023 | shopper | year_ago | 0.052 | 0.033 to 0.080 | 169 | no | no |
| credit_deadline_2025 | enthusiast | prior_3_months | 0.015 | -0.017 to 0.052 | 436 | yes | no |
| credit_deadline_2025 | enthusiast | year_ago | 0.023 | -0.010 to 0.057 | 860 | yes | no |
| credit_deadline_2025 | shopper | prior_3_months | 0.018 | -0.033 to 0.069 | 139 | yes | no |
| credit_deadline_2025 | shopper | year_ago | 0.023 | -0.026 to 0.072 | 183 | yes | no |

Figure: `v2/outputs/framing_event_study.png`.

### 5.3 Interrupted series

The level shift is the post-event change in the monthly share of the pre-specified barrier. The interval resamples months.

| Event | Audience | Level shift | Month-bootstrap interval | Months | Includes 0 |
|---|---|---:|---|---:|---|
| ira_2022 | enthusiast | 0.031 | -0.064 to 0.058 | 17 | yes |
| ira_2022 | shopper | -0.011 | -0.047 to 0.015 | 19 | yes |
| tesla_price_cut_2023 | enthusiast | 0.040 | -0.177 to 0.090 | 19 | yes |
| tesla_price_cut_2023 | shopper | 0.077 | -0.042 to 0.168 | 19 | yes |
| nacs_2023 | enthusiast | -0.055 | -0.209 to 0.005 | 19 | yes |
| nacs_2023 | shopper | 0.012 | -0.058 to 0.055 | 19 | yes |
| credit_deadline_2025 | enthusiast | 0.009 | -0.042 to 0.041 | 18 | yes |
| credit_deadline_2025 | shopper | 0.005 | -0.064 to 0.062 | 19 | yes |

Figure: `v2/outputs/framing_its_level.png`. That figure draws HC1 intervals. The table above is the month bootstrap.

What these rows can support, even before validation, is a statement about the labeled file: which contrasts still exclude zero after the baseline months were filled, and which ones do not. What they cannot support is a sentence about the effect of the Act, the price cut, the connector announcement, or the credit deadline.

Filling the four thin enthusiast months changed these zero-inclusion decisions (design-aware bootstrap, unvalidated):

- ira_2022, enthusiast, prior_3_months: 0.049 (0.002 to 0.096, excluded zero) became 0.042 (-0.001 to 0.087, includes zero). Baseline posts 202 to 897.
- ira_2022, enthusiast, year_ago: 0.064 (0.024 to 0.106, excluded zero) became 0.041 (-0.003 to 0.088, includes zero). Baseline posts 23 to 485.
- credit_deadline_2025, enthusiast, year_ago: 0.038 (0.002 to 0.076, excluded zero) became 0.023 (-0.010 to 0.057, includes zero). Baseline posts 72 to 860.
- Interrupted series, nacs_2023, enthusiast: -0.097 (-0.271 to -0.001, excluded zero) became -0.055 (-0.209 to 0.005, includes zero).

The before-and-after tables are in `v2/outputs/FINDINGS.md`.

### 5.4 Shopper composition

The full shopper file is a poor stand-in for people who are shopping. Two slices are in `v2/outputs/framing_sensitivity_shares.csv` and `framing_sensitivity_contrasts.csv`: r/carbuying alone, and posts the stage label marks as pre-purchase (learning, comparing, evaluating, or buying). On the unvalidated labels, price and dealer friction are high in r/carbuying, and the ownership share there is still above the enthusiast share. The January 2023 price rise in the full shopper file does not reappear in the pre-purchase slice. Those sentences are about the model's stage label, which is itself unvalidated.

## 6. Validation plan

The sheet is `v2/annotation/annotation_sheet.csv`: 800 posts (401 enthusiast, 399 shopper), excerpts of 500 characters, code cells empty. The draw oversamples each barrier toward 40 positives per audience, then 25 posts per audience in each of the four event windows, then year-band cells. The practice rows in `annotation_preview.csv` are excluded. Model labels are in `annotation_model_key.csv`, not on the sheet. The codebook is `v2/annotation/CODEBOOK.md`.

`v2/score_agreement.py` reads two coder files and writes Cohen's kappa, Krippendorff's alpha, precision, recall, and F1 of the model against the items both coders marked and agreed on, and a Rogan-Gladen correction of the prevalence. It was run on synthetic labels only. It does not invent a coder. Disagreements are counted and then left out of the gold set. Blanks are missing, not a third category.

The rule already fixed in `v2/paper/introduction.md` is: if the lower bound of the confidence interval on barrier kappa is below 0.60, add 200 adjudicated posts before a substantive claim, and if it is still too low, hold the Transportation Research Part D submission.

No second annotator has been named. This draft does not contact or recruit anyone.

## 7. Limitations

English-language Reddit is not the US household population. The enthusiast file is a cluster sample of submissions, not the 603,435-post census, and the round 6 fill is still a cap of 40 posts per week rather than every submission in those four months. The shopper file is keyword-filtered, so it over-represents owners and supporters talking about Tesla, Rivian, or Hyundai. Comments are absent; Ruan and Lv's corpus is mostly comments. Arctic Shift does not guarantee that the archive is complete. The Academic Torrents license field is empty. Other news happens in the same months as the four shocks, so a level shift is not an effect. The zero-shot model is small. A larger model might label differently; that is an open decision, not a result. Labels have not been checked by a person.

## 8. Open decisions

These are unanswered. The draft does not pick them.

1. **Second annotator.** The sheet is built for two coders. Nobody has been named, and nobody should be contacted from this repository.
2. **Stage taxonomy.** One ownership stage, or Ryan's onboarding split. The taxonomy PDF is not in the repository. The sheet's onboarding column can be collapsed either way.
3. **Model size.** The labels in this draft are the small open CPU model. A paid API or a GPU is not used.
4. **Comments.** Submissions only. A comments pull is a later decision. The cars comments file in the per-subreddit torrent is about 2.2 GB compressed and was not downloaded.
5. **Upvote filter.** No score threshold is applied. A sensitivity with score at least 2 has not been run.

Framing A is locked. Shopper text stays on the Academic Torrents dumps. Those are not open decisions.

## 9. What has to happen before submission

One page. Nothing below is optional if the target is Transportation Research Part D.

1. Name the second annotator and the adjudication rule. Do not start coding with one person and call it agreement.
2. Freeze the stage taxonomy (one ownership code, or ownership plus onboarding) before coding starts, and point the codebook at that choice.
3. Both annotators code `v2/annotation/annotation_sheet.csv` against `v2/annotation/CODEBOOK.md`. The practice file stays out of the agreement calculation.
4. Run `v2/score_agreement.py` on the two files and the model key. Report kappa and alpha per barrier, stage, and stance, with the confidence intervals the script writes.
5. If the lower bound on barrier kappa is below 0.60, draw and code 200 more posts before any substantive sentence. If it is still below that line, do not send the barrier claims to this journal.
6. Adjudicate disagreements. Score the model against the agreed items. Apply the prevalence correction to the corpus rates, and replace section 5 with those corrected rates.
7. Decide, in writing, whether a larger open model or a paid API will relabel the corpus. If it will, relabel and repeat steps 4–6. If it will not, say that the instrument is the small CPU model.
8. Decide comments and the upvote filter. If either is adopted, rerun the contrasts and show the sensitivity. If either is declined, say so in the limitations.
9. Resolve every `[CITATION NEEDED]` mark. Add the survey papers already in use. Do not add a DOI that has not been opened. Open the primary entries for Cohen, Krippendorff, Rogan-Gladen, and VADER, or drop the method names that lack an entry.
10. Replace the figures with the validated series. Keep the design-aware intervals. State the sample sizes, the missing months, and that the estimates are descriptive.
11. Write the data-availability and ethics paragraphs the journal asks for: public posts, no attempt to identify users, dumps not republished, license field empty.
12. Check the journal's current author guide for length, highlights, and reference style, then format. This draft does not guess those limits.

## References

Ashby, C., Weir, D., and Fussey, P. (2025). Understanding public views on electric vehicle charging: A thematic analysis. *Transportation Research Interdisciplinary Perspectives*, 29, 101325. https://doi.org/10.1016/j.trip.2025.101325

Gilardi, F., Alizadeh, M., and Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *PNAS*, 120(30), e2305016120. https://doi.org/10.1073/pnas.2305016120

Ha, S., Marchetto, D. J., Dharur, S., and Asensio, O. I. (2021). Topic classification of electric vehicle consumer experiences with transformer-based deep learning. *Patterns*, 2, February 2021. https://doi.org/10.1016/j.patter.2020.100195

Ruan, T., and Lv, Q. (2022). Public perception of electric vehicles on reddit over the past decade. *Communications in Transportation Research*, 2, 100070. https://doi.org/10.1016/j.commtr.2022.100070

Ruan, T., and Lv, Q. (2023). Public perception of electric vehicles on Reddit and Twitter: A cross-platform analysis. *Transportation Research Interdisciplinary Perspectives*, 21, 100872. https://doi.org/10.1016/j.trip.2023.100872

Yu, Q., Que, T., Cushing, L. J., Pierce, G., Shen, K., Kejriwal, M., Yao, Y., and Zhu, Y. (2025). Equity and reliability of public electric vehicle charging stations in the United States. *Nature Communications*. https://doi.org/10.1038/s41467-025-60091-y

Watchful1. (2026). Subreddit comments/submissions 2005-06 to 2025-12 [Data set]. Academic Torrents. Infohash `3e3f64dee22dc304cdd2546254ca1f8e8ae542b4`. The license field on the torrent is empty.

Event dates, as used in `v2/research_design.md`: Pub. L. 117-169; Reuters, 13 January 2023, https://www.reuters.com/business/autos-transportation/tesla-cuts-prices-electric-vehicles-us-market-2023-01-13/; Reuters, 25 May 2023, https://www.reuters.com/business/autos-transportation/tesla-ford-ceos-talk-evs-twitter-forum-2023-05-25/; General Motors news release, 8 June 2023, https://investor.gm.com/news-releases/news-release-details/general-motors-doubles-down-commitment-unified-charging-standard/; Pub. L. 119-21; 26 U.S.C. § 30D; IRS, https://www.irs.gov/clean-vehicle-tax-credits.

Hutto and Gilbert (2014), Cohen, Krippendorff, and Rogan and Gladen: [CITATION NEEDED].

Survey literature on adoption barriers: [CITATION NEEDED].
