# Research design: public views of electric vehicles on Reddit

This memo is for Daisy. It recommends one paper, states what would be new relative to the papers below, and specifies the codebook and the human-validation plan. The counts for the corpus pulled in this round live in `v2/data_card/DATA_CARD.md`, which the downloader writes from the API responses. Nothing in that card is estimated by hand.

The original pipeline under `src/` is unchanged. `v2/` is a pilot of the measurement the paper needs.

## 1. What has to change for a paper

The repository today names topics. A Gemini call labels each topic, and a keyword list then assigns the topic to a purchase stage. There is no document-level label for barriers, stance, or journey stage, and no human check on any label.

That is the gap a Transportation Research Part D reviewer, or an applied NLP reviewer, will treat as fatal. Three published strands already cover "topics and sentiment about EVs on social media":

- Ruan and Lv (2022) collected 3,437,917 Reddit posts (274,979 submissions and 3,162,938 comments) from January 2011 through December 2020. They fit a 20-topic Mallet LDA model through Gensim and compared LIWC, VADER, and AFINN, reporting the VADER series. Replication package: <https://github.com/TaoRuan-Campus/EVReddit>. The article is in *Communications in Transportation Research*, volume 2, article 100070, not Transportation Research Part D. <https://doi.org/10.1016/j.commtr.2022.100070>
- Ruan and Lv (2023) repeat that Reddit corpus and add 7,383,327 tweets over the same decade, again with Gensim LDA (20 topics on each platform) and AFINN, VADER, and LIWC. *Transportation Research Interdisciplinary Perspectives* 21, 100872. <https://doi.org/10.1016/j.trip.2023.100872>
- Ashby, Weir, and Fussey (2025) build a charging-specific thematic structure from just under 100,000 tweets with BERTopic, a large language model, and manual review, then describe sentiment and volume by theme. *Transportation Research Interdisciplinary Perspectives* 29, 101325. <https://doi.org/10.1016/j.trip.2025.101325>

Two further papers set the methodological bar, on charging reviews rather than Reddit:

- Yu, Que, Cushing, Pierce, Shen, Kejriwal, Yao, and Zhu (2025) run an LLM over 470,142 charging-station reviews. Two research assistants annotated sentiment on 3,188 reviews (Cohen's κ = 0.81). GPT-3.5 accuracy was 86.2% and GPT-4 accuracy was 87.1% on that sentiment task. For problem categories the human inter-rater κ was 0.71 and LLM accuracy was 0.77 to 0.82. *Nature Communications*, published 18 June 2025. <https://doi.org/10.1038/s41467-025-60091-y>
- Ha, Marchetto, Dharur, and Asensio (2021) train a transformer classifier on expert-annotated charging reviews. *Patterns* 2 (February 2021). <https://doi.org/10.1016/j.patter.2020.100195>. This is the template for a supervised baseline once a gold set exists. Accuracy figures are not repeated here because the publisher page could not be retrieved in this session.

Gilardi, Alizadeh, and Kubli (2023) is the citation for using an LLM as an annotator, and for the obligation that comes with it. On four samples (n = 6,183) zero-shot ChatGPT beat crowd workers by about 25 percentage points on average. *PNAS* 120(30), e2305016120. <https://doi.org/10.1073/pnas.2305016120>

A second gap is the clock. The Ruan corpora stop in December 2020. The policy dates that now organize US EV buying are later: the Inflation Reduction Act (Pub. L. 117-169, 16 August 2022), Tesla's price cuts reported on 13 January 2023, Ford's charging agreement of 25 May 2023 and GM's NACS announcement of 8 June 2023, and the end of the federal clean-vehicle credit for vehicles acquired after 30 September 2025 (Pub. L. 119-21, 4 July 2025; 26 U.S.C. § 30D; IRS, <https://www.irs.gov/clean-vehicle-tax-credits>).

## 2. The paper

Framing A is the paper. Daisy locked it in on 17 October 2026. The draft introduction and the hypotheses are in `v2/paper/introduction.md`. Two earlier alternatives are an appendix at the end of this memo. They are not competing papers.

**Question.** How do the EV purchase barriers people discuss on Reddit — price, charging, battery, range, incentives, dealer friction, and environment — shift around the policy and market shocks in section 6, and do enthusiast subreddits and mainstream car-shopper subreddits move together?

The shocks are the Inflation Reduction Act (16 August 2022), Tesla's US price cuts reported on 13 January 2023, Ford's and GM's moves onto Tesla's charging connector (25 May and 8 June 2023), and the end of the federal clean-vehicle credit for vehicles acquired after 30 September 2025.

The enthusiast subreddits are r/electricvehicles, r/teslamotors, r/BoltEV, r/Rivian, r/Ioniq5, r/KiaEV6, and r/evcharging. The shopper subreddits are r/cars, r/Autos, and r/carbuying. A subreddit name is not a journey stage. Owner versus shopper language is a contrast inside this paper, using the document-level stage code, not a separate paper.

Research questions:

1. Which of the seven barriers are common from 2015 through September 2026, in enthusiast subreddits and in the shopper subreddits, and which rise or fall?
2. At the coded event months, does the share of posts touching incentives, price, or charging differ from the prior three months and from the same months a year earlier? Submission volume is reported beside those shares. It is not itself a barrier result.
3. Do the shopper subreddits put more of their barrier talk on price and dealer friction, and less on the ownership stage, than the enthusiast subreddits?

**What would be new.** Document-level barrier, stage, and stance labels through 2026, tied to dates Ruan and Lv could not observe, with the human-validation design in section 5. The shopper subreddits are in the design, not only brand communities. Ashby, Weir, and Fussey already combine BERTopic and an LLM, but on tweets, about charging only, and the published abstract describes manual review of the theme list rather than a double-coded test of document labels. Yu et al. do that test, on charging-station reviews and a different question. The zero-shot model, and the comparison with VADER, LDA, and the keyword stage rule, are the methods that make the barrier claims believable. They are not a separate contribution standing in for the human labels.

**Risk.** Volume changes are not effects of the policy. Several of these dates sit on a rising market. The paper has to say "discussion moved" and show a year-ago comparison, not "the IRA caused" anything. If the double-coded kappa on barriers is too low to support a substantive claim, the same materials support a shorter methods paper and the Transportation Research Part D submission should wait. That decision comes after the annotation.

The market-funnel workbook can stay in the repository as an exploratory tool. It is not the contribution.

## 3. What the methods section has to carry

The language-model labels are the instrument. Section 5 is the validation plan: two annotators, 800 posts, Cohen's kappa per field, then macro-F1 of the model against the adjudicated gold. VADER is the sentiment baseline Ruan and Lv reported. LDA is a descriptive topic baseline. The keyword stage rule, both the old substring version and the word-boundary version, is there so the paper can show the false matches. None of those three is the barrier measure.

The dealer-friction label is on when the model score clears 0.5 or a markup, fee, or inventory lexicon matches, and it is held off for ordinary service and for a charger that merely sits at a dealer. That rule was checked on 18 invented sentences (`v2/fixtures/dealer_sentences.csv`). It has not been checked on corpus posts by a second person.

## 4. Draft codebook

Codes are applied to the submission (title plus body). A post may receive more than one barrier. Stage and stance are single-label. Illustrations below are written for the codebook. They are not quotations from the corpus.

### 4.1 Barriers (multi-label)

| Code | Apply when the post is about | Do not apply when |
|---|---|---|
| `charging_access` | Getting a charge: home electrical work, apartments, public stations, queues, broken or blocked chargers, connector type | The only energy mention is gasoline prices |
| `range` | Whether the car can cover a trip, winter loss, range anxiety | Battery degradation with no mention of distance |
| `price_tco` | Purchase price, lease, insurance, depreciation, electricity versus fuel, repair bills | A tax credit or rebate, which is `incentives_policy` even if it changes the price |
| `incentives_policy` | Tax credit, rebate, eligibility, the IRA, a credit deadline, a mandate | A private discount with no public incentive |
| `battery_reliability` | Degradation, replacement, recall, fire, "will the battery last" | Range on a single trip with no durability claim |
| `dealer_friction` | Dealer markup, inventory, wait, pressure, the buying transaction | Ordinary service of a car already owned, which is ownership content and usually `battery_reliability` or no barrier |
| `environment` | Emissions, mining, the grid, whether the car is actually cleaner | — |
| `none` | No barrier above. Exclusive: if any other barrier is present, `none` is off | — |

### 4.2 Journey stage (single-label)

| Code | The writer is |
|---|---|
| `awareness` | Learning what an EV is, or whether to consider one at all |
| `consideration` | Comparing models, brands, or EV versus a gasoline car, without a near-term purchase |
| `evaluation` | Working a budget, a credit, a commute, or a charger for a purchase they are close to making |
| `purchase` | At a dealer, placing an order, taking delivery |
| `ownership` | Already has the vehicle: living with it, service, road trips in it |
| `unclear` | The stage cannot be told, or the post is not about a personal vehicle decision |

Ryan's branch splits ownership into onboarding and long-term ownership. The reference taxonomy PDF is not in the repository. Until Daisy decides, ownership stays one code. Annotators can tick an optional `onboarding` flag (first months, learning the car) without making it a seventh stage.

### 4.3 Stance toward EVs (single-label)

| Code | Meaning |
|---|---|
| `support` | The writer recommends EVs, defends them, or describes a good outcome they attribute to the technology |
| `oppose` | The writer argues against buying or subsidizing EVs |
| `mixed` | Both, or a complaint that leaves the writer's position genuinely unresolved |
| `no_stance` | A factual or procedural post, or a brand complaint that does not generalize to EVs |

A complaint about one broken charger is not `oppose` by itself. `oppose` requires a claim that reaches beyond the incident.

### 4.4 Worked illustrations (invented)

- "The credit made the lease cheaper than my old Civic, picking it up Saturday." Barriers: `incentives_policy`, `price_tco`. Stage: `purchase`. Stance: `support`.
- "Third floor, no garage, the public station was ICE'd again." Barrier: `charging_access`. Stage: `ownership` if they have the car, otherwise `evaluation`. Stance: `mixed` unless they draw a broader conclusion.
- "Every EV is a coal car." Barrier: `environment`. Stage: `unclear`. Stance: `oppose`.

## 5. Validation plan

**Who.** Two annotators. Daisy is one. The second is a labmate, who will code soon. This project did not contact or recruit anyone. The second coder should not be the person who wrote the prompt, if that person is also an author, or the conflict should be stated.

**Practice.** The 25-row `v2/annotation/annotation_preview.csv` is for discussion. It is excluded from kappa. The codebook is revised once, then frozen.

**Gold sample.** 800 submissions, stratified by community proxy (brand community, general EV community, mainstream car subreddit) and by year band (2015–2017, 2018–2020, 2021–2023, 2024–latest). `v2/run_analysis.py` writes the draw to `v2/annotation/annotation_sample.csv` with empty code columns. If the downloaded sample has fewer than 800 kept posts, the sheet contains every kept post and the shortfall is reported. Authors are not in the sheet.

**Why 800.** The review's range was 500 to 1,000. Yu et al. used 3,188 for a binary sentiment task. Eight hundred is enough to estimate kappa on stage and stance and on each barrier, and small enough for two people to finish. If the lower bound of the confidence interval on barrier kappa is below 0.60, add 200 adjudicated posts before a substantive claim. That rule should be fixed before looking at the model scores.

The draw in `v2/annotation/annotation_sheet.csv` is that 800, with the practice rows in `annotation_preview.csv` held out. Round 7 redrew it for the 6-stage funnel and the 8 topic families. Fifty rows are flagged `calibration=yes` and are excluded from kappa. The codebook is `v2/annotation/CODEBOOK.md`. The taxonomy note is `v2/docs/reference_taxonomy_6stage_8family.pdf`. Model predictions are in `annotation_model_key.csv`, not on the sheet. `v2/score_agreement.py` computes Cohen's kappa, Krippendorff's alpha, model-versus-human precision/recall/F1, and a Rogan-Gladen prevalence correction once two coder files exist. It leaves the calibration rows out of the main table. No coder has filled the sheet.

**Statistics.**

- Stage and stance: Cohen's kappa, plus percent agreement. Report the marginal distribution, because a rare `oppose` class makes kappa hard to interpret.
- Each barrier: Cohen's kappa on the binary present/absent judgment.
- A single multi-label summary: mean of the per-barrier kappas, and Krippendorff's alpha if the two coders leave different items blank.
- Disagreements are adjudicated in a meeting. Items still unresolved are coded `unclear` or barrier-absent. They are not forced.

**Model comparison, after adjudication.**

- Zero-shot NLI (the open model in `v2/ev_v2/annotate.py`) and, if a key exists later, one API model at temperature 0.
- VADER, mapped to support / oppose / no_stance with compound thresholds of +0.05 and −0.05. VADER is not a barrier classifier. It is the sentiment baseline Ruan and Lv reported.
- The word-boundary keyword stage rule in `v2/ev_v2/funnel.py`, and the old substring rule, so the paper can show the false-friend rate.
- LDA topic probabilities as a descriptive baseline, not as a barrier measure.
- Metrics: macro-F1 and Cohen's kappa against the adjudicated gold, per field, with a confusion matrix. A fine-tuned encoder is a later baseline once the gold set exists. It is not part of this pilot.

**What this round does not claim.** No human labels were collected. NLI shares in `v2/outputs/` are a first look at the instrument, not a result about the public.

## 6. Events, described rather than "identified"

| ID | Date used | Window for monthly volume | Source |
|---|---|---|---|
| `ira_2022` | 16 Aug 2022 | 2022-08 | Pub. L. 117-169, cited in the amendment notes to 26 U.S.C. § 30D, <https://uscode.house.gov/us/usc/119-118/t26/s30D/> |
| `tesla_price_cut_2023` | 13 Jan 2023 | 2023-01 | Reuters, 13 Jan 2023, <https://www.reuters.com/business/autos-transportation/tesla-cuts-prices-electric-vehicles-us-market-2023-01-13/> |
| `nacs_2023` | 25 May 2023 (Ford) and 8 Jun 2023 (GM) | 2023-05 and 2023-06 | Reuters, 25 May 2023, <https://www.reuters.com/business/autos-transportation/tesla-ford-ceos-talk-evs-twitter-forum-2023-05-25/>; GM release, 8 Jun 2023, <https://investor.gm.com/news-releases/news-release-details/general-motors-doubles-down-commitment-unified-charging-standard/> |
| `credit_law_2025` | 4 Jul 2025 | 2025-07 | Pub. L. 119-21, 4 Jul 2025, amendment notes to 26 U.S.C. § 30D; IRS FAQ, <https://www.irs.gov/newsroom/faqs-for-modification-of-sections-25c-25d-25e-30c-30d-45l-45w-and-179d-under-public-law-119-21-139-stat-72-july-4-2025-commonly-known-as-the-one-big-beautiful-bill-obbb> |
| `credit_deadline_2025` | 30 Sep 2025 | 2025-09 and 2025-10 | 26 U.S.C. § 30D as amended; IRS, <https://www.irs.gov/clean-vehicle-tax-credits> |

For each window the pilot reports the mean monthly submission count against the prior three months and against the same months one year earlier. Both comparisons are descriptive. The text sample adds a mean VADER score in the window, with the sample size printed next to it.

## 7. Data and ethics

The downloader uses the Arctic Shift API (`https://arctic-shift.photon-reddit.com`), whose maintainer documents the endpoints and states that there is no uptime guarantee. Core subreddits are counted in full, one month at a time. Mainstream subreddits have a keyword count census and a partial text pull. Details, including what the text pull did not finish, are in the data card.

The pull on 16 October 2026 returned a count for every core subreddit-month from January 2015 through October 2026. October 2026 is partial. Complete months sum to 603,435 submissions. A later pass counted keyword submissions in r/cars, r/Autos, and r/carbuying from January 2015 through September 2026: 16,993 posts, and every requested month returned a count. That 16,993 is an API aggregate, not a text file. Shopper text is extracted from the Academic Torrents zstd dumps by `v2/extract_from_dumps.py`. The dump filter is a word-boundary match, so its count is not required to equal 16,993. The data card records both. The local API text file used for the pilot labels has 8,866 submissions and is still mostly the earlier core sample plus early r/cars posts about Tesla.

Usernames are kept only in the gitignored local file, so the bot filter can run. The committed excerpt and the annotation sheet do not contain usernames. The pull does not attempt to identify anyone. Short excerpts in the annotation file are for coding, not for republication in the paper.

## 8. Decisions

Framing A is locked, and the paper is built on the funnel. The shopper-text route is the dumps. Submissions only, until a later decision. The stage system is the 6-stage funnel in `v2/docs/reference_taxonomy_6stage_8family.pdf`, with onboarding kept separate from long-term ownership. Each post gets one stage, one or more of the eight topic families, the seven barrier aspects, and a stance. The instrument for the corpus is an open instruction model run on Daisy's own GPU (`v2/llm_label.py`, `v2/RUN_ON_MY_GPU.md`). The small NLI model stays as the baseline. No paid API.

Still open:

1. The labmate's name is not written in the repo. They will code `v2/annotation/annotation_sheet.csv`. Nobody was contacted from this project.
2. Which checkpoint to run first. The note recommends Qwen2.5-7B-Instruct in 4-bit on the 16 GB card. Llama-3.1-8B-Instruct and Qwen2.5-14B-Instruct are alternatives. 14B may not fit.
3. Whether comments should be pulled. The original papers are mostly comments. This design censuses submissions. The per-subreddit comments files exist in the same torrent and are much larger (r/cars comments alone are about 2.16 GB compressed).
4. Whether a score filter (submission score at least 2) should be a sensitivity check. It is not applied by default.

The discontinuity and the author-transition counts in round 7 are scaffolds on the small-model labels. They are replaced when the instruction-model labels, and then the human labels, exist.

## 9. Discontinuity and stage transitions

Four cutoffs, local linear in time, bandwidth chosen from 30, 60, 90, and 180 days, a 7-day donut, and placebos 180 days away:

- IRA signing, 16 August 2022, outcome incentives.
- IRA credit rules for vehicles placed in service after 31 December 2022, cutoff 1 January 2023, outcome incentives. Source: Pub. L. 117-169 and 26 U.S.C. § 30D, the same statute already cited for the August 2022 enactment. This cutoff is twelve days before the Tesla price-cut date, so the two fits are not separate experiments.
- Tesla price cuts, 13 January 2023, outcome price.
- Credit end, 30 September 2025, outcome incentives.

Stage transitions use authors with at least two posts. The small model cannot split onboarding from long-term ownership, so the ladder stops at one ownership rung. Repeat-author coverage is the feasibility number. A week sample is a hard place to see the same person twice.

## Appendix. Framings that are not the paper

These two were the alternatives considered before Framing A was locked. They stay here so a reader of the memo can see what was set aside. They are not written up in `v2/paper/`.

### Owner language versus shopper language, as its own paper

The contrast is real, and it is inside Framing A as research question 3. It is not the lead question. A subreddit is a weak proxy: r/teslamotors contains shoppers, and r/cars contains owners. The contrast is only credible if journey stage is a document label that humans have checked. The pilot shows why a keyword stage rule is not that label. Substring matching treats `ev` inside `every`, and `vs` inside `evs`, as evidence of the Consideration stage.

### A methods paper on the language model alone

On a double-coded sample, a zero-shot model can be compared with VADER, LDA, and the keyword stage rule. That comparison is the methods section of Framing A. It cannot be finished without the human labels, and it is not a substitute for the barrier question. If barrier kappa is too low, this is the shorter paper to write instead, as section 2 already says.
