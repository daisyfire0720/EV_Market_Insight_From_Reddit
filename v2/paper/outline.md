# Electric-vehicle talk on Reddit, 2015–2026

Working title. The lead question is locked. The paper asks how the purchase barriers people discuss — price, charging, battery, range, incentives, dealer friction, and environment — shift around the Inflation Reduction Act (August 2022), Tesla's price cuts (January 2023), NACS adoption (May–June 2023), and the end of the federal clean-vehicle credit for vehicles acquired after 30 September 2025. Enthusiast subreddits are compared with mainstream car-shopper subreddits (r/cars, r/Autos, r/carbuying). The language model and its human validation are the methods. The drafted introduction and hypotheses are `introduction.md`. Transportation Research Part D shape. Placeholders in braces are filled only from `v2/outputs/` after a run. They are not targets.

## Abstract

Public discussion is one of the few records of how buyers talk about electric vehicles between surveys. Earlier Reddit work measured topics and dictionary sentiment through 2020. This paper measures adoption barriers, purchase-journey stage, and stance at the level of the post, checks those labels against two human annotators, and describes how the barrier mix sits around four later shocks: the Inflation Reduction Act (16 August 2022), Tesla's price cuts reported on 13 January 2023, the Ford and GM moves onto Tesla's charging connector (25 May and 8 June 2023), and the end of the US clean-vehicle credit for vehicles acquired after 30 September 2025. The corpus is submissions from dedicated EV communities, counted month by month in the Arctic Shift archive, plus keyword-matched submissions from r/cars, r/Autos, and r/carbuying. {One sentence on the human kappa, once it exists.} {One sentence on which barriers dominate, and whether incentive talk rises in the credit windows, once the validated labels exist.} The pilot reported here is the measurement harness and the volume series. It is not yet the validated result.

## 1. Introduction

Electric-vehicle policy assumes that price, charging, and information are what keep households out. Those assumptions are usually tested with surveys of intenders. Surveys miss the language people use when they are not being asked, and they miss the months in which a credit, a price cut, or a charging-standard announcement actually lands.

Reddit is a public record of that language. Ruan and Lv (2022) showed that it is large enough to recover topics and a sentiment trend, using 3,437,917 posts from 2011 through 2020, a 20-topic LDA model, and VADER. Ruan and Lv (2023) showed the Reddit pattern is not identical to Twitter. Ashby, Weir, and Fussey (2025) showed that a topic model plus a language model can organize charging talk on Twitter, with manual review of the themes. None of those papers measures, at the post and with a double-coded check, the barrier a writer is raising, the stage of the purchase they are in, and whether they are for or against the technology, through the policy dates that now define the US market.

The paper leads with those barriers, and with whether shopper subreddits and enthusiast subreddits move together. The language model is the instrument that assigns the barriers. It is not yet validated. Owner-versus-shopper language is a contrast inside the paper, using the stage label on each post, not a separate study. None of the comparisons around the policy dates are causal estimates. The hypotheses are written out in `introduction.md`.

### Contributions

1. A document-level codebook of adoption barriers, journey stage, and stance, with two-annotator reliability reported the way Yu et al. (2025) report it for charging reviews.
2. A comparison of that instrument with the tools the earlier papers actually used (VADER, LDA) and with the keyword stage rule in the existing repository, including the false matches that rule produces.
3. A monthly submission series for the main EV subreddits from 2015 through the latest archived month, read against the 2022–2025 policy and price dates, and a contrast between brand communities, general EV forums, and mainstream car subreddits.

## 2. Related work

Three clusters, kept short.

**Adoption barriers in surveys.** Price, range, charging, and information are the usual list. Cite the survey literature Daisy already uses. Do not invent a canonical list beyond the codebook in `v2/research_design.md`.

**Social media as a record of EV opinion.** Ruan and Lv (2022, 2023) as above. Ashby, Weir, and Fussey (2025): BERTopic and an LLM on just under 100,000 charging tweets. State the platform, the stopping date, and the absence of a post-level barrier label validated like Yu et al.

**Language models as annotators.** Gilardi, Alizadeh, and Kubli (2023) on when zero-shot labeling beats crowd workers, and the requirement to validate. Yu et al. (2025): 470,142 reviews, 3,188 double-coded, sentiment κ = 0.81, problem-category human κ = 0.71, LLM accuracy 0.77–0.82. Ha, Marchetto, Dharur, and Asensio (2021) as the supervised-transformer alternative once labels exist.

## 3. Data

Source, range, subreddits, and filters: `v2/data_card/DATA_CARD.md`. Repeat in the paper only the figures that card computes.

- Counts: Arctic Shift, because Pushshift no longer updates and the official Reddit API requires credentials this project does not have.
- Shopper text: the per-subreddit zstd dumps (infohash `3e3f64dee22dc304cdd2546254ca1f8e8ae542b4`, through the end of 2025), streamed by `v2/extract_from_dumps.py`. January–September 2026 shopper text is only in the monthly `RS_*.zst` files, which were not downloaded here. The data card has the lengths, the checksum notes, and the empty license field.
- Unit: submissions. Comments are a documented next pull. Say so. Ruan and Lv's mass is mostly comments, so the comparison of levels is not like-for-like.
- Core census: r/electricvehicles, r/teslamotors, r/BoltEV, r/Rivian, r/Ioniq5, r/KiaEV6, r/evcharging, monthly, 2015 through the latest complete month. The in-progress month is reported separately.
- Mainstream posts: r/cars, r/Autos, r/carbuying, keyword-filtered. The API count of 16,993 is not the dump count. The dump filter is a word-boundary match and will not be retuned to force the two totals together. The keyword list omits bare `ev`.
- Text sample design: deterministic 7-day window per subreddit-quarter, plus a fixed week around each event. State that this is a cluster sample.
- Cleaning: deleted bodies removed; bot accounts removed with a token rule that does not drop surnames containing "bot" or names starting with "auto". Counts of removals: {from `run_manifest.json`}.
- Ethics: public posts, hashed usernames in the released excerpt, no attempt to identify users.

{Table 1. Submissions per subreddit and year. Source: `v2/data_card/yearly_post_counts.csv`.}

## 4. Methods

### 4.1 Codebook

Point to section 4 of `v2/research_design.md`. Barriers are multi-label. Stage and stance are single-label. Ownership is one stage until the onboarding split is decided.

### 4.2 Human annotation

800 posts, two annotators, practice set excluded from kappa, Cohen's kappa per field, adjudication rules. Not yet executed. The sheet is `v2/annotation/annotation_sample.csv`.

### 4.3 Models

- VADER (Hutto and Gilbert 2014, via the `vaderSentiment` package), the dictionary Ruan and Lv reported.
- LDA with a probability for every document (sklearn, 8 topics in the pilot, seed 42). Eight topics is a pilot setting for a submission sample, not a claim that 8 is the right number for the census. Ruan and Lv used 20 on a corpus three orders of magnitude larger. Per-document maxima are stored. The original BERTopic configuration left those probabilities empty.
- Zero-shot natural-language inference with `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` on CPU, unless `run_manifest.json` records a different revision. No paid API. Barrier and stage hypotheses use "This text is about {}." Stance uses "This text expresses {}." Those templates were picked by running five invented codebook sentences, not by fitting the corpus. A smaller DistilBERT MNLI checkpoint did not separate the same sentences.
- Dealer friction adds a lexicon on top of that score. A markup, doc fee, market adjustment, or dealer inventory wait is on even when the model scores the sentence as price. Ordinary service and a charger that merely sits at a dealer are left off when the lexicon does not match. The comparison is `outputs/dealer_rule_before_after.csv`, on invented sentences, not on the corpus.
- Keyword stage baseline, word-boundary and substring, using the five stage lists already in the repository.

### 4.4 Event windows

Windows and sources are listed in the research-design memo. For each, report mean monthly submissions in the window, divided by the prior three months, and divided by the same months a year earlier. Print the number of months observed. Do not call the ratio an effect.

### 4.5 What the pilot already runs

`v2/run_analysis.py` runs cleaning, VADER, both keyword rules, LDA, the volume ratios, and the NLI subset. Human kappa is a later table.

## 5. Results

The figures in this section are from the pilot run on 16 October 2026. They are copied from `v2/data_card/` and `v2/outputs/`. Volume is the census of submissions. Labels are the text sample. Nothing here is a validated claim about "the public." Replace this section when the gold set exists.

### 5.1 How much people posted

Complete months, January 2015 through September 2026, contain 603,435 submissions. October 2026 was partial on the day of the pull (2,440 submissions through 16 October). Year totals, from `yearly_post_counts.csv`: 2015 9,074; 2016 19,902; 2017 24,753; 2018 40,089; 2019 60,455; 2020 46,208; 2021 62,926; 2022 83,904; 2023 73,127; 2024 75,232; 2025 65,916; January–September 2026 41,849.

The same nine months a year earlier were higher: January–September 2024 was 57,104 and January–September 2025 was 52,616. r/teslamotors was 82.3% of the seven-subreddit total in 2015 and 7.8% in January–September 2026. r/Rivian, r/Ioniq5, and r/KiaEV6 account for most of the shift. 2020 is below 2019 (46,208 against 60,455). Within 2020, January was 5,185 submissions and April was 2,579.

Figure: `outputs/monthly_submissions.png`.

### 5.2 Volume around the five dates

From `outputs/event_volume.csv`. Ratios are mean monthly submissions in the window divided by the comparison mean. They are not effects.

| Window | Event-month mean | / prior 3 months | / same months, prior year |
|---|---:|---:|---:|
| August 2022 (IRA enactment) | 8,080 | 1.076 | 1.570 |
| January 2023 (Tesla price cuts) | 6,154 | 0.987 | 0.929 |
| May–June 2023 (Ford and GM on NACS) | 5,930.5 | 1.032 | 0.808 |
| July 2025 (credit-ending law) | 5,567 | 0.991 | 0.903 |
| September–October 2025 (acquisition deadline) | 4,976.5 | 0.908 | 0.825 |

August 2022 is only 8% above the prior three months. The 1.57 ratio against August 2021 sits on the multi-year rise and should not be read as the Act's effect. The 2023 and 2025 windows are flat or lower than both comparisons. In these seven subreddits, the credit deadline does not show up as a surge in how many posts were made.

### 5.3 Keyword counts outside the EV subreddits

The 17 October 2026 aggregate counted 16,993 submissions in r/cars (14,800), r/Autos (1,684), and r/carbuying (509) from January 2015 through September 2026. Every requested month returned a count. 2026 in that table is January–September. That figure is an API count. The same day's dump extract, from the per-subreddit submissions files through the end of 2025, kept 13,976 posts (r/cars 12,210, r/Autos 1,428, r/carbuying 338) after the word-boundary filter. The torrent has no 2026 posts. The API rows for January–September 2026 are 485 of the 16,993 (r/cars 305, r/Autos 38, r/carbuying 142). The year-by-year comparison is `data_card/dump_vs_api_counts.csv`. The two filters are not the same matcher, and the dump column was not adjusted to equal the API column. The earlier API text pull is not this extract.

### 5.4 Sentiment in the text sample

The 17 October run cleaned 8,243 posts (8,866 downloaded; 23 bot rows; 600 shorter than five tokens). Mean VADER compound is 0.196 in the brand communities (n = 4,469), 0.204 in the general EV communities (n = 1,933), and 0.071 in the mainstream rows that survived cleaning (n = 1,841). The mainstream mean is mostly early r/cars posts about Tesla, not a sample of the 16,993.

### 5.5 What the keyword rule was measuring

On these 8,243 posts, substring matching put 40.7% in Consideration. Word-boundary matching put 23.7% in Consideration and 54.6% in Unclear. Figure: `outputs/funnel_match_comparison.png`.

### 5.6 Topics

Eight-topic LDA, seed 42, refit on the larger text file. Every document has a probability; the mean of the maximum is 0.629 and the minimum is 0.125. The largest assigned topic (1,655 documents) has top words `tesla, model, autopilot, musk`. Words are in `outputs/lda_topics.csv`. These are not human-authored titles.

### 5.7 First look at the NLI labels

Unvalidated. Model `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`, revision `262ae02f29173eec1c250f90804dc7edc677dcff`, on all 8,243 cleaned posts. Dealer friction uses the score or the lexicon. Shares: price or total cost 0.171, charging access 0.142, battery reliability 0.099, driving range 0.064, environmental impact 0.059, dealer or buying friction 0.029, incentives 0.027. Stage: owning 0.473, no clear stage 0.238, comparing 0.206. Stance: support 0.601, mixed 0.236, no stance 0.148, opposition 0.015 (124 posts).

On 18 invented sentences the original dealer score matched the intended label 10 times. The lexicon rule matched 18 times. A reworded dealer label matched 11 times and was not adopted. On the 8,243 posts the lexicon turned 16 additional posts on and the service/charging guard turned 22 off (241 above 0.5 before the rule, 235 after). The corpus effect is small. The markup sentences are where the rule matters. None of this is human-validated.

### 5.8 Communities

Do not read the mainstream VADER mean or the NLI stance-by-community table as a shopper-versus-owner result. The text is early r/cars Tesla posts plus a few dozen posts from r/Autos and r/carbuying. The count census is the part that covers 2015–2026.

## 6. Policy reading

Only after section 5 is validated. The reading the journal will want, if the labels hold:

- If incentive language spikes in July 2025 and September 2025 and not in the year-ago months, the credit deadline is visible in public talk, which is relevant to how sunset clauses are communicated.
- If charging remains a large share in brand communities after the NACS announcements, a connector agreement is not, by itself, the end of charging talk.
- If mainstream car subreddits are more price-heavy and less ownership-heavy than brand communities, incentive design is being heard by shoppers only to the extent those shoppers are in the mainstream sample. Say the sample limit in the same paragraph.

Delete any bullet the tables do not support.

## 7. Limitations

- Submissions, not comments. The earlier Reddit papers are mostly comments.
- The text file is a designed sample of weeks, not a simple random sample of posts. Volume uses the census. Label shares use the sample.
- Mainstream subreddits are keyword-filtered. The API text pull is incomplete. The dump extract is the text source, and it still has to be read against the API count rather than substituted for it. See the data card. The per-subreddit torrent ends in December 2025.
- Arctic Shift does not guarantee completeness. Very new posts have placeholder scores.
- Zero-shot labels on a small CPU model are a lower bound on what a larger model might do, not an upper bound. The human comparison is the check either way.
- Event ratios are descriptive. Other news happens in the same months.
- English-language Reddit is not the US household population. Ruan and Lv already note that these platforms skew young.
- The original local corpus (Pushshift extracts on Daisy's machine) was not in the repository and was not used. Results here do not describe that corpus.

## 8. Conclusion

One paragraph. Repeat the three contributions. State what a reader should believe (the volume series; the failure mode of substring stage rules; the design of the validation) and what they should not believe yet (any sentence about which barrier "the public" holds).

## References to include, each opened or resolved against Crossref during this round

- Ashby, C., Weir, D., and Fussey, P. (2025). Understanding public views on electric vehicle charging: A thematic analysis. *Transportation Research Interdisciplinary Perspectives*, 29, 101325. <https://doi.org/10.1016/j.trip.2025.101325>
- Gilardi, F., Alizadeh, M., and Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *PNAS*, 120(30), e2305016120. <https://doi.org/10.1073/pnas.2305016120>
- Ha, S., Marchetto, D. J., Dharur, S., and Asensio, O. I. (2021). Topic classification of electric vehicle consumer experiences with transformer-based deep learning. *Patterns*, 2, February 2021. <https://doi.org/10.1016/j.patter.2020.100195>
- Hutto, C. J., and Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. Cited via Ruan and Lv (2022); the paper text names this source. Do not add a DOI that was not opened.
- Ruan, T., and Lv, Q. (2022). Public perception of electric vehicles on reddit over the past decade. *Communications in Transportation Research*, 2, 100070. <https://doi.org/10.1016/j.commtr.2022.100070>
- Ruan, T., and Lv, Q. (2023). Public perception of electric vehicles on Reddit and Twitter: A cross-platform analysis. *Transportation Research Interdisciplinary Perspectives*, 21, 100872. <https://doi.org/10.1016/j.trip.2023.100872>
- Yu, Q., Que, T., Cushing, L. J., Pierce, G., Shen, K., Kejriwal, M., Yao, Y., and Zhu, Y. (2025). Equity and reliability of public electric vehicle charging stations in the United States. *Nature Communications*, published 18 June 2025. <https://doi.org/10.1038/s41467-025-60091-y>

Statute and news sources for the event dates are listed in `v2/research_design.md` and should be repeated in the paper's data section, not only in a footnote.
