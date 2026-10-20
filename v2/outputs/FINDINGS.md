# Pilot findings (unvalidated)

Labels in this file are not human-coded. Barrier, stage, and stance shares are model output. Volume ratios are descriptive, not effects.

The paper's question is locked. It asks how the barriers people discuss — price, charging, battery, range, incentives, dealer friction, and environment — shift around the Inflation Reduction Act (August 2022), Tesla's price cuts (January 2023), NACS adoption (May–June 2023), and the end of the federal clean-vehicle credit on 30 September 2025, and whether enthusiast subreddits and shopper subreddits (r/cars, r/Autos, r/carbuying) move together. The hypotheses are in `v2/paper/introduction.md`. The Framing A section, when it is present, is a first look at model labels. It is not a validated test.

Run time (UTC): `2026-10-17T04:57:44.217791+00:00`.

## Core submission census

Complete months through the month before `2026-10` sum to **603435** submissions in the seven core subreddits. `2026-10` is excluded from that sum when it is the in-progress month of the pull.

| Year | Submissions |
|---|---:|
| 2015 | 9074 |
| 2016 | 19902 |
| 2017 | 24753 |
| 2018 | 40089 |
| 2019 | 60455 |
| 2020 | 46208 |
| 2021 | 62926 |
| 2022 | 83904 |
| 2023 | 73127 |
| 2024 | 75232 |
| 2025 | 65916 |
| 2026 | 41849 |

## Mainstream keyword counts

EV-keyword submissions in r/cars, r/Autos, and r/carbuying. The query and the failure list are in the data card. A failed month is not a zero.

| Subreddit | Year | Posts | Months ok | Partial | Failed |
|---|---:|---:|---:|---:|---:|
| r/Autos | 2015 | 118 | 12 | 0 | 0 |
| r/Autos | 2016 | 169 | 12 | 0 | 0 |
| r/Autos | 2017 | 136 | 12 | 0 | 0 |
| r/Autos | 2018 | 152 | 12 | 0 | 0 |
| r/Autos | 2019 | 248 | 12 | 0 | 0 |
| r/Autos | 2020 | 187 | 12 | 0 | 0 |
| r/Autos | 2021 | 190 | 12 | 0 | 0 |
| r/Autos | 2022 | 163 | 12 | 0 | 0 |
| r/Autos | 2023 | 115 | 12 | 0 | 0 |
| r/Autos | 2024 | 80 | 12 | 0 | 0 |
| r/Autos | 2025 | 88 | 12 | 0 | 0 |
| r/Autos | 2026 | 38 | 9 | 0 | 0 |
| r/carbuying | 2015 | 1 | 12 | 0 | 0 |
| r/carbuying | 2016 | 0 | 12 | 0 | 0 |
| r/carbuying | 2017 | 2 | 12 | 0 | 0 |
| r/carbuying | 2018 | 4 | 12 | 0 | 0 |
| r/carbuying | 2019 | 4 | 12 | 0 | 0 |
| r/carbuying | 2020 | 22 | 12 | 0 | 0 |
| r/carbuying | 2021 | 18 | 12 | 0 | 0 |
| r/carbuying | 2022 | 22 | 12 | 0 | 0 |
| r/carbuying | 2023 | 51 | 12 | 0 | 0 |
| r/carbuying | 2024 | 84 | 12 | 0 | 0 |
| r/carbuying | 2025 | 159 | 12 | 0 | 0 |
| r/carbuying | 2026 | 142 | 9 | 0 | 0 |
| r/cars | 2015 | 849 | 12 | 0 | 0 |
| r/cars | 2016 | 1318 | 12 | 0 | 0 |
| r/cars | 2017 | 1428 | 12 | 0 | 0 |
| r/cars | 2018 | 1719 | 12 | 0 | 0 |
| r/cars | 2019 | 1788 | 12 | 0 | 0 |
| r/cars | 2020 | 1252 | 12 | 0 | 0 |
| r/cars | 2021 | 1677 | 12 | 0 | 0 |
| r/cars | 2022 | 1373 | 12 | 0 | 0 |
| r/cars | 2023 | 1285 | 12 | 0 | 0 |
| r/cars | 2024 | 1163 | 12 | 0 | 0 |
| r/cars | 2025 | 643 | 12 | 0 | 0 |
| r/cars | 2026 | 305 | 9 | 0 | 0 |

Those counts are archive aggregates from the API (16,993 in total). They are not a text file.

## Shopper text from the dumps (counts only)

Unvalidated as a statement about barriers. This is a count of posts the extractor kept, not a coded result. On 17 October 2026 the per-subreddit submission files (infohash `3e3f64dee22dc304cdd2546254ca1f8e8ae542b4`) were streamed in full: 1,077,267 lines, 0 bad JSON lines. The word-boundary filter kept **13,976** posts dated 2015-01-01 through 2025-12-28: r/cars 12,210, r/Autos 1,428, r/carbuying 338. The files end in 2025, so they do not contain the 485 API posts from January–September 2026. Against the API aggregate of 16,993 (2015 through September 2026), the dump total is lower in every subreddit, and the gap is wider in later years of r/cars (102 posts in 2015, 328 in 2022). The API search matches tokens the word-boundary rule skips, including `ioniq` inside `ioniq5`. The two columns were not forced together. Year table: `v2/data_card/dump_vs_api_counts.csv`. The zero-shot table later in this file is the earlier pilot. The Framing A section is the combined run: the enthusiast sample plus the cleaned dump posts.

The earlier search pass finished r/cars for the keyword tesla from January 2015 through April 2017, except July 2015 and June 2016, which were deferred after rate limits. That API text pull is not the dump extract.

## Event windows

Mean monthly submissions in the seven core subreddits.

| Window | Mean | Versus prior 3 months | Versus a year earlier |
|---|---:|---:|---:|
| ira_2022 | 8080.0 | 1.076 | 1.570 |
| tesla_price_cut_2023 | 6154.0 | 0.987 | 0.929 |
| nacs_2023 | 5930.5 | 1.032 | 0.808 |
| credit_law_2025 | 5567.0 | 0.991 | 0.903 |
| credit_deadline_2025 | 4976.5 | 0.908 | 0.825 |

## Text sample

Cleaned posts in this run: **8243** (input 8866, bots removed 23, short or empty 600).

| Community proxy | Posts | Mean VADER compound |
|---|---:|---:|
| brand_community | 4469 | 0.196 |
| general_ev_community | 1933 | 0.204 |
| mainstream_car | 1841 | 0.071 |

Posts by subreddit after cleaning: r/cars 1802, r/electricvehicles 1336, r/teslamotors 1209, r/BoltEV 1071, r/Rivian 803, r/Ioniq5 714, r/KiaEV6 672, r/evcharging 597, r/Autos 30, r/carbuying 9.

## Keyword stage rule

Substring matching assigned 40.7% of cleaned posts to Consideration. Word-boundary matching assigned 23.7% to Consideration and 54.6% to Unclear.

## Zero-shot labels (unvalidated)

Model `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` revision `262ae02f29173eec1c250f90804dc7edc677dcff`, n=8243, threshold 0.5. Dealer friction is on when the NLI score clears the threshold or the dealer-friction lexicon matches. Seconds this process spent inside the labeler: 550.1. That clock includes posts already in the checkpoint, which return without a forward pass.

| Barrier | On | Share |
|---|---:|---:|
| vehicle price or total cost | 1406 | 0.171 |
| charging access | 1173 | 0.142 |
| battery reliability | 812 | 0.099 |
| driving range | 530 | 0.064 |
| environmental impact | 487 | 0.059 |
| dealer or buying friction | 235 | 0.029 |
| purchase incentives or tax credits | 223 | 0.027 |

### Stage

| Label | n | Share |
|---|---:|---:|
| owning an electric vehicle | 3897 | 0.473 |
| no clear purchase stage | 1961 | 0.238 |
| comparing electric vehicle options | 1696 | 0.206 |
| buying an electric vehicle | 362 | 0.044 |
| learning about electric vehicles | 201 | 0.024 |
| evaluating cost or fit before buying | 126 | 0.015 |

### Stance

| Label | n | Share |
|---|---:|---:|
| support for electric vehicles | 4954 | 0.601 |
| mixed or uncertain view of electric vehicles | 1942 | 0.236 |
| no stance on electric vehicles | 1223 | 0.148 |
| opposition to electric vehicles | 124 | 0.015 |

Dealer-rule reasons (why the dealer bit is on or off):

- `off`: 7986
- `nli`: 207
- `suppressed_service_or_charging_location`: 22
- `lexicon`: 16
- `nli+lexicon`: 12

## What this does not show

These labels have not been checked by a second coder. Event ratios are not effects. A mainstream count can be complete while the text pull for that month is still short. Arctic Shift does not guarantee that the archive is complete.

<!-- framing-a-start -->
## Framing A labels (unvalidated)

This section is model output. No human has coded these posts. The intervals in the table below treat posts as independent, which is too narrow for a file built from sampled weeks. The week and month block intervals are in the round 5 section under this one. A difference here is not an effect of the statute or the price cut.

Enthusiast rows are the text sample from the seven EV subreddits. Shopper rows are the dump extract from r/cars, r/Autos, and r/carbuying, including the January–September 2026 monthly files when that extract has been merged. A probable bot, or a post under five tokens, is dropped before labeling, which is why these counts are smaller than the raw extract.

### Shares in the whole labeled file

| Measure | Enthusiast share | n | Shopper share | n | Shopper minus enthusiast | 95% interval |
|---|---:|---:|---:|---:|---:|---|
| charging access | 0.173 | 9044 | 0.034 | 13783 | -0.139 | -0.147 to -0.130 |
| driving range | 0.070 | 9044 | 0.060 | 13783 | -0.011 | -0.017 to -0.004 |
| vehicle price or total cost | 0.175 | 9044 | 0.175 | 13783 | -0.000 | -0.010 to 0.010 |
| purchase incentives or tax credits | 0.030 | 9044 | 0.014 | 13783 | -0.016 | -0.020 to -0.012 |
| battery reliability | 0.112 | 9044 | 0.041 | 13783 | -0.071 | -0.079 to -0.064 |
| dealer or buying friction | 0.033 | 9044 | 0.027 | 13783 | -0.006 | -0.011 to -0.002 |
| environmental impact | 0.066 | 9044 | 0.047 | 13783 | -0.019 | -0.025 to -0.013 |
| owning an electric vehicle | 0.387 | 9044 | 0.629 | 13783 | 0.242 | 0.229 to 0.254 |
| support for electric vehicles | 0.513 | 9044 | 0.815 | 13783 | 0.302 | 0.290 to 0.315 |

### Event windows

Each contrast is the share in the event month or months minus the share in the prior three months, and minus the same months a year earlier. The pre-specified barrier is incentives for August 2022 and for September–October 2025, price for January 2023, and charging for May–June 2023. The other barriers are reported in `framing_event_contrasts.csv`.

| Event | Audience | Baseline | Event share | Baseline share | Difference | 95% interval | Event n | Baseline n |
|---|---|---|---:|---:|---:|---|---:|---:|
| credit_deadline_2025 / purchase incentives or tax credits | enthusiast | prior_3_months | 0.052 | 0.037 | 0.015 | -0.015 to 0.045 | 328 | 436 |
| credit_deadline_2025 / purchase incentives or tax credits | enthusiast | year_ago | 0.052 | 0.029 | 0.023 | -0.004 to 0.049 | 328 | 860 |
| credit_deadline_2025 / purchase incentives or tax credits | shopper | prior_3_months | 0.061 | 0.043 | 0.018 | -0.040 to 0.076 | 98 | 139 |
| credit_deadline_2025 / purchase incentives or tax credits | shopper | year_ago | 0.061 | 0.038 | 0.023 | -0.032 to 0.078 | 98 | 183 |
| ira_2022 / purchase incentives or tax credits | enthusiast | prior_3_months | 0.064 | 0.021 | 0.042 | 0.010 to 0.075 | 236 | 897 |
| ira_2022 / purchase incentives or tax credits | enthusiast | year_ago | 0.064 | 0.023 | 0.041 | 0.007 to 0.075 | 236 | 485 |
| ira_2022 / purchase incentives or tax credits | shopper | prior_3_months | 0.022 | 0.022 | 0.001 | -0.034 to 0.036 | 89 | 277 |
| ira_2022 / purchase incentives or tax credits | shopper | year_ago | 0.022 | 0.000 | 0.022 | -0.008 to 0.053 | 89 | 95 |
| nacs_2023 / charging access | enthusiast | prior_3_months | 0.155 | 0.224 | -0.069 | -0.134 to -0.004 | 554 | 201 |
| nacs_2023 / charging access | enthusiast | year_ago | 0.155 | 0.153 | 0.002 | -0.037 to 0.041 | 554 | 809 |
| nacs_2023 / charging access | shopper | prior_3_months | 0.076 | 0.048 | 0.028 | -0.018 to 0.074 | 184 | 269 |
| nacs_2023 / charging access | shopper | year_ago | 0.076 | 0.024 | 0.052 | 0.008 to 0.097 | 184 | 169 |
| tesla_price_cut_2023 / vehicle price or total cost | enthusiast | prior_3_months | 0.212 | 0.140 | 0.073 | 0.020 to 0.125 | 292 | 845 |
| tesla_price_cut_2023 / vehicle price or total cost | enthusiast | year_ago | 0.212 | 0.052 | 0.161 | 0.087 to 0.234 | 292 | 58 |
| tesla_price_cut_2023 / vehicle price or total cost | shopper | prior_3_months | 0.247 | 0.150 | 0.097 | 0.014 to 0.179 | 146 | 260 |
| tesla_price_cut_2023 / vehicle price or total cost | shopper | year_ago | 0.247 | 0.120 | 0.127 | 0.036 to 0.217 | 146 | 125 |

### Interrupted series

Monthly shares from 12 months before the first event month through 6 months after, weighted by the number of labeled posts. The post coefficient is the level shift at the event month, after a linear pre-trend and a change in slope. Months with fewer than 5 labeled posts are left out. A missing month is skipped, not entered as zero.

| Event | Audience | Level shift | 95% interval | Months | Posts | Post months | Status |
|---|---|---:|---|---:|---:|---:|---|
| credit_deadline_2025 | enthusiast | 0.009 | -0.012 to 0.031 | 18 | 2297 | 7 | ok |
| credit_deadline_2025 | shopper | 0.005 | -0.051 to 0.060 | 19 | 1300 | 7 | ok |
| ira_2022 | enthusiast | 0.031 | 0.007 to 0.055 | 17 | 3152 | 7 | ok |
| ira_2022 | shopper | -0.011 | -0.037 to 0.015 | 19 | 1917 | 7 | ok |
| nacs_2023 | enthusiast | -0.055 | -0.116 to 0.006 | 19 | 3413 | 7 | ok |
| nacs_2023 | shopper | 0.012 | -0.036 to 0.060 | 19 | 1734 | 7 | ok |
| tesla_price_cut_2023 | enthusiast | 0.040 | -0.032 to 0.111 | 19 | 3343 | 7 | ok |
| tesla_price_cut_2023 | shopper | 0.077 | 0.018 to 0.136 | 19 | 1820 | 7 | ok |

Figures: `framing_barrier_shares_monthly.png`, `framing_event_study.png`, `framing_its_level.png`. Event-study points are the share in that relative month minus the share in the month before the event (`framing_event_study.csv`).
<!-- framing-a-end -->

<!-- round5-start -->
## Round 5 checks (unvalidated)

The labels are still the small open model. No person has coded the annotation sheet. Intervals in this section resample the design: sampled weeks for the enthusiast file, and calendar months for shoppers. A window built from fewer months than the design asks for is flagged. None of these differences is an effect of the statute or the price cut. If the round 6 baseline fill is already in the label file, these tables are the filled file. The before-and-after comparison is in the round 6 section.

### Annotation sample

The sheet has 800 submissions (401 enthusiast, 399 shopper). Text excerpts were filled for 800 of them. The size is the 800 already set in the research design, so two annotators can finish it. Inside that 800, each audience is asked for 40 posts on each barrier before the rest of the sample is drawn, then 25 posts in each of the four event windows, then a year-band remainder. Posts added only to fill the year bands are dropped first if the layers run past 800. The practice file of 25 is excluded.

| Audience | Code | Posts in the audience draw | Positives in the draw |
|---|---|---:|---:|
| enthusiast | incentives_policy | 401 | 51 |
| enthusiast | dealer_friction | 401 | 47 |
| enthusiast | environment | 401 | 65 |
| enthusiast | range | 401 | 59 |
| enthusiast | battery_reliability | 401 | 63 |
| enthusiast | charging_access | 401 | 66 |
| enthusiast | price_tco | 401 | 99 |
| enthusiast | none | 401 | 131 |
| shopper | incentives_policy | 399 | 43 |
| shopper | dealer_friction | 399 | 42 |
| shopper | environment | 399 | 54 |
| shopper | range | 399 | 64 |
| shopper | battery_reliability | 399 | 48 |
| shopper | charging_access | 399 | 45 |
| shopper | price_tco | 399 | 99 |
| shopper | none | 399 | 142 |

### Block-bootstrap intervals

The difference is the event-window share minus the baseline share, same point estimate as the independent-post interval. The new interval is the 2.5 and 97.5 percentiles of 2,000 cluster resamples. Enthusiast blocks are sampled weeks. Shopper blocks are calendar months when both sides have at least two months; otherwise the reported interval resamples subreddit-months and the one-month window is flagged.

| Event | Audience | Baseline | Difference | Independent interval | Block interval | Block | Thin window | Few blocks | Event n | Baseline n |
|---|---|---|---:|---|---|---|---|---|---:|---:|
| ira_2022 | enthusiast | prior_3_months | 0.042 | 0.010 to 0.075 | -0.001 to 0.087 | week_block | no | no | 236 | 897 |
| ira_2022 | enthusiast | year_ago | 0.041 | 0.007 to 0.075 | -0.003 to 0.088 | week_block | no | no | 236 | 485 |
| ira_2022 | shopper | prior_3_months | 0.001 | -0.034 to 0.036 | -0.041 to 0.147 | subreddit_month | no | no | 89 | 277 |
| ira_2022 | shopper | year_ago | 0.022 | -0.008 to 0.053 | 0.000 to 0.167 | subreddit_month | no | no | 89 | 95 |
| tesla_price_cut_2023 | enthusiast | prior_3_months | 0.073 | 0.020 to 0.125 | -0.026 to 0.162 | week_block | no | no | 292 | 845 |
| tesla_price_cut_2023 | enthusiast | year_ago | 0.161 | 0.087 to 0.234 | 0.035 to 0.270 | week_block | no | no | 292 | 58 |
| tesla_price_cut_2023 | shopper | prior_3_months | 0.097 | 0.014 to 0.179 | 0.025 to 0.840 | subreddit_month | no | no | 146 | 260 |
| tesla_price_cut_2023 | shopper | year_ago | 0.127 | 0.036 to 0.217 | -0.737 to 0.874 | subreddit_month | no | no | 146 | 125 |
| nacs_2023 | enthusiast | prior_3_months | -0.069 | -0.134 to -0.004 | -0.233 to 0.064 | week_block | no | no | 554 | 201 |
| nacs_2023 | enthusiast | year_ago | 0.002 | -0.037 to 0.041 | -0.071 to 0.081 | week_block | no | no | 554 | 809 |
| nacs_2023 | shopper | prior_3_months | 0.028 | -0.018 to 0.074 | -0.008 to 0.068 | month_block | no | no | 184 | 269 |
| nacs_2023 | shopper | year_ago | 0.052 | 0.008 to 0.097 | 0.033 to 0.080 | month_block | no | no | 184 | 169 |
| credit_deadline_2025 | enthusiast | prior_3_months | 0.015 | -0.015 to 0.045 | -0.017 to 0.052 | week_block | no | no | 328 | 436 |
| credit_deadline_2025 | enthusiast | year_ago | 0.023 | -0.004 to 0.049 | -0.010 to 0.057 | week_block | no | no | 328 | 860 |
| credit_deadline_2025 | shopper | prior_3_months | 0.018 | -0.040 to 0.076 | -0.033 to 0.069 | month_block | no | no | 98 | 139 |
| credit_deadline_2025 | shopper | year_ago | 0.023 | -0.032 to 0.078 | -0.026 to 0.072 | month_block | no | no | 98 | 183 |

Where the two intervals disagree about 0: ira_2022 / enthusiast / prior_3_months (the block interval includes 0); ira_2022 / enthusiast / year_ago (the block interval includes 0); tesla_price_cut_2023 / enthusiast / prior_3_months (the block interval includes 0); tesla_price_cut_2023 / shopper / year_ago (the block interval includes 0); nacs_2023 / enthusiast / prior_3_months (the block interval includes 0).
Some block intervals exclude 0 and are still too wide to read as a precise shift, because a side has only a few clusters: tesla_price_cut_2023 / shopper / prior_3_months (0.025 to 0.840); tesla_price_cut_2023 / shopper / year_ago (-0.737 to 0.874).

### Interrupted series, month bootstrap

Each draw resamples the months in the 12-before / 6-after window and refits the level shift. The HC1 interval is the one from the original weighted regression.

| Event | Audience | Level shift | HC1 interval | Month-bootstrap interval | HC1 excludes 0 | Bootstrap excludes 0 | Months |
|---|---|---:|---|---|---|---|---:|
| credit_deadline_2025 | enthusiast | 0.009 | -0.012 to 0.031 | -0.042 to 0.041 | no | no | 18 |
| credit_deadline_2025 | shopper | 0.005 | -0.051 to 0.060 | -0.064 to 0.062 | no | no | 19 |
| ira_2022 | enthusiast | 0.031 | 0.007 to 0.055 | -0.064 to 0.058 | yes | no | 17 |
| ira_2022 | shopper | -0.011 | -0.037 to 0.015 | -0.047 to 0.015 | no | no | 19 |
| nacs_2023 | enthusiast | -0.055 | -0.116 to 0.006 | -0.209 to 0.005 | no | no | 19 |
| nacs_2023 | shopper | 0.012 | -0.036 to 0.060 | -0.058 to 0.055 | no | no | 19 |
| tesla_price_cut_2023 | enthusiast | 0.040 | -0.032 to 0.111 | -0.177 to 0.090 | no | no | 19 |
| tesla_price_cut_2023 | shopper | 0.077 | 0.018 to 0.136 | -0.042 to 0.168 | yes | no | 19 |

### Shopper composition

Pre-purchase means the stage label is learning, comparing, evaluating, or buying. Owning and “no clear purchase stage” are left out of that slice. r/carbuying is the third shopper subreddit on its own.

| Slice | Measure | Share | n | Shopper minus enthusiast | 95% interval |
|---|---|---:|---:|---:|---|
| all_shopper | charging access | 0.034 | 13783 | -0.139 | -0.147 to -0.130 |
| carbuying | charging access | 0.064 | 468 | -0.109 | -0.132 to -0.085 |
| enthusiast | charging access | 0.173 | 9044 |  |  to  |
| pre_purchase | charging access | 0.040 | 4521 | -0.133 | -0.143 to -0.124 |
| all_shopper | dealer or buying friction | 0.027 | 13783 | -0.006 | -0.011 to -0.002 |
| carbuying | dealer or buying friction | 0.239 | 468 | 0.206 | 0.167 to 0.245 |
| enthusiast | dealer or buying friction | 0.033 | 9044 |  |  to  |
| pre_purchase | dealer or buying friction | 0.027 | 4521 | -0.006 | -0.012 to 0.000 |
| all_shopper | owning an electric vehicle | 0.629 | 13783 | 0.242 | 0.229 to 0.254 |
| carbuying | owning an electric vehicle | 0.502 | 468 | 0.115 | 0.068 to 0.161 |
| enthusiast | owning an electric vehicle | 0.387 | 9044 |  |  to  |
| pre_purchase | owning an electric vehicle | 0.000 | 4521 | -0.387 | -0.397 to -0.377 |
| all_shopper | purchase incentives or tax credits | 0.014 | 13783 | -0.016 | -0.020 to -0.012 |
| carbuying | purchase incentives or tax credits | 0.120 | 468 | 0.089 | 0.060 to 0.119 |
| enthusiast | purchase incentives or tax credits | 0.030 | 9044 |  |  to  |
| pre_purchase | purchase incentives or tax credits | 0.014 | 4521 | -0.016 | -0.021 to -0.011 |
| all_shopper | support for electric vehicles | 0.815 | 13783 | 0.302 | 0.290 to 0.315 |
| carbuying | support for electric vehicles | 0.709 | 468 | 0.196 | 0.154 to 0.239 |
| enthusiast | support for electric vehicles | 0.513 | 9044 |  |  to  |
| pre_purchase | support for electric vehicles | 0.818 | 4521 | 0.305 | 0.289 to 0.320 |
| all_shopper | vehicle price or total cost | 0.175 | 13783 | -0.000 | -0.010 to 0.010 |
| carbuying | vehicle price or total cost | 0.806 | 468 | 0.631 | 0.594 to 0.667 |
| enthusiast | vehicle price or total cost | 0.175 | 9044 |  |  to  |
| pre_purchase | vehicle price or total cost | 0.213 | 4521 | 0.038 | 0.024 to 0.052 |

Primary event contrasts inside the slices (independent-post intervals, because several slices are a few dozen posts):

| Slice | Event | Baseline | Difference | 95% interval | Event n | Baseline n |
|---|---|---|---:|---|---:|---:|
| all_shopper | credit_deadline_2025 | prior_3_months | 0.018 | -0.040 to 0.076 | 98 | 139 |
| all_shopper | credit_deadline_2025 | year_ago | 0.023 | -0.032 to 0.078 | 98 | 183 |
| carbuying | credit_deadline_2025 | prior_3_months | 0.058 | -0.136 to 0.252 | 18 | 46 |
| carbuying | credit_deadline_2025 | year_ago | -0.064 | -0.351 to 0.222 | 18 | 13 |
| pre_purchase | credit_deadline_2025 | prior_3_months | 0.062 | -0.021 to 0.146 | 32 | 46 |
| pre_purchase | credit_deadline_2025 | year_ago | 0.030 | -0.064 to 0.125 | 32 | 62 |
| all_shopper | ira_2022 | prior_3_months | 0.001 | -0.034 to 0.036 | 89 | 277 |
| all_shopper | ira_2022 | year_ago | 0.022 | -0.008 to 0.053 | 89 | 95 |
| carbuying | ira_2022 | prior_3_months | -0.500 | -0.990 to -0.010 | 1 | 4 |
| carbuying | ira_2022 | year_ago | 0.000 | 0.000 to 0.000 | 1 | 1 |
| pre_purchase | ira_2022 | prior_3_months | 0.053 | -0.033 to 0.139 | 32 | 106 |
| pre_purchase | ira_2022 | year_ago | 0.062 | -0.021 to 0.146 | 32 | 34 |
| all_shopper | nacs_2023 | prior_3_months | 0.028 | -0.018 to 0.074 | 184 | 269 |
| all_shopper | nacs_2023 | year_ago | 0.052 | 0.008 to 0.097 | 184 | 169 |
| carbuying | nacs_2023 | prior_3_months | 0.143 | -0.116 to 0.402 | 7 | 6 |
| carbuying | nacs_2023 | year_ago | 0.143 | -0.116 to 0.402 | 7 | 3 |
| pre_purchase | nacs_2023 | prior_3_months | -0.001 | -0.063 to 0.061 | 72 | 93 |
| pre_purchase | nacs_2023 | year_ago | 0.026 | -0.029 to 0.081 | 72 | 64 |
| all_shopper | tesla_price_cut_2023 | prior_3_months | 0.097 | 0.014 to 0.179 | 146 | 260 |
| all_shopper | tesla_price_cut_2023 | year_ago | 0.127 | 0.036 to 0.217 | 146 | 125 |
| carbuying | tesla_price_cut_2023 | prior_3_months | 0.250 | -0.174 to 0.674 | 2 | 4 |
| carbuying | tesla_price_cut_2023 | year_ago | 0.000 | 0.000 to 0.000 | 2 | 4 |
| pre_purchase | tesla_price_cut_2023 | prior_3_months | -0.039 | -0.170 to 0.091 | 46 | 94 |
| pre_purchase | tesla_price_cut_2023 | year_ago | 0.054 | -0.078 to 0.186 | 46 | 51 |

On r/carbuying alone (n=468), price is 0.806 and dealer friction is 0.239. Both sit above the enthusiast file (gaps 0.631 and 0.206, and both intervals exclude 0). Ownership is still 0.502 and support is 0.709. That ownership share is still above the enthusiast file. The event-window cells in r/carbuying are mostly single-digit counts, so those contrasts are not a check of the shocks.

The pre-purchase slice has n=4521 and, by construction, no ownership posts. Price is 0.213 (gap 0.038, interval 0.024 to 0.052). Dealer friction is 0.027 (gap -0.006, interval -0.012 to 0.000). Support is still 0.818.

The pre-purchase price contrast for January 2023 versus the prior three months is -0.039 (-0.170 to 0.091, event n=46, baseline n=94).

The codebook, the blank sheet, and the model key are in `v2/annotation/`. `v2/score_agreement.py` scores two coder files when they exist. It was checked on synthetic labels only.
<!-- round5-end -->

<!-- round6-start -->
## Round 6 thin baselines (unvalidated)

The labels are still the small open model. No person has coded them. This round filled only the enthusiast months Round 5 flagged, under a cap of **3360** posts (40 per subreddit-window) and **90** minutes of CPU labeling. The 282,348-post dense pull was not started. Differences are still not effects.

Pulled 2887 submissions, kept 2642 after the bot and length filter, and appended 2642 new rows (20185 to 22827). Labeling took 17.55 minutes. Windows requested: 84.

| Month | New labeled posts |
|---|---:|
| 2021-08 | 462 |
| 2022-06 | 695 |
| 2022-12 | 697 |
| 2024-09 | 788 |

### What moved in the block-bootstrap contrasts

3 primary reported intervals changed whether zero is inside the bootstrap interval.

| Event | Audience | Baseline | Before | After | Baseline n before | Baseline n after | Zero before | Zero after |
|---|---|---|---:|---:|---:|---:|---|---|
| ira_2022 | enthusiast | prior_3_months | 0.049 (0.002 to 0.096) | 0.042 (-0.001 to 0.087) | 202 | 897 | excludes 0 | includes 0 |
| ira_2022 | enthusiast | year_ago | 0.064 (0.024 to 0.106) | 0.041 (-0.003 to 0.088) | 23 | 485 | excludes 0 | includes 0 |
| credit_deadline_2025 | enthusiast | year_ago | 0.038 (0.002 to 0.076) | 0.023 (-0.010 to 0.057) | 72 | 860 | excludes 0 | includes 0 |

Primary reported intervals that still exclude zero after the fill:

- tesla_price_cut_2023 / enthusiast / year_ago: 0.161 (0.035 to 0.270, baseline n=58).
- tesla_price_cut_2023 / shopper / prior_3_months: 0.097 (0.025 to 0.840, baseline n=260).
- nacs_2023 / shopper / year_ago: 0.052 (0.033 to 0.080, baseline n=169).

Point estimates that moved by more than 0.02 and did not change their relation to zero:

- nacs_2023 / enthusiast / year_ago: 0.068 to 0.002 (baseline n 114 to 809).

### What moved in the interrupted series

1 primary reported interval changed whether zero is inside the bootstrap interval.

| Event | Audience | Before | After | Posts before | Posts after | Zero before | Zero after |
|---|---|---:|---:|---:|---:|---|---|
| nacs_2023 | enthusiast | -0.097 (-0.271 to -0.001) | -0.055 (-0.209 to 0.005) | 2021 | 3413 | excludes 0 | includes 0 |

### Enthusiast window coverage

| Event | Window | Posts before | Posts after | Missing after |
|---|---|---:|---:|---|
| ira_2022 | event | 236 | 236 |  |
| ira_2022 | prior_3_months | 202 | 897 |  |
| ira_2022 | year_ago | 23 | 485 |  |
| tesla_price_cut_2023 | event | 292 | 292 |  |
| tesla_price_cut_2023 | prior_3_months | 148 | 845 |  |
| tesla_price_cut_2023 | year_ago | 58 | 58 |  |
| nacs_2023 | event | 554 | 554 |  |
| nacs_2023 | prior_3_months | 201 | 201 |  |
| nacs_2023 | year_ago | 114 | 809 |  |
| credit_deadline_2025 | event | 328 | 328 |  |
| credit_deadline_2025 | prior_3_months | 436 | 436 |  |
| credit_deadline_2025 | year_ago | 72 | 860 |  |

No enthusiast event, prior-three, or year-ago window is missing a designed month after the fill.

The manuscript draft that uses these tables is `v2/paper/manuscript.md`. Every share in it is preliminary until the annotation sheet is coded.
<!-- round6-end -->
