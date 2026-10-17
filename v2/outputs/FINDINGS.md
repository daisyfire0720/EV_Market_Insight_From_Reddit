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

This section is model output. No human has coded these posts. A confidence interval here is uncertainty in the labeled sample under a simple independent-posts calculation. It is not an effect of the statute or the price cut, and it is too narrow for the enthusiast file, which is a sample of weeks rather than every submission.

Enthusiast rows are the text sample from the seven EV subreddits. Shopper rows are the dump extract from r/cars, r/Autos, and r/carbuying. The shopper files stop at the end of 2025. A probable bot, or a post under five tokens, is dropped before labeling, which is why these counts are smaller than the 13,976-post extract and the raw enthusiast file.

### Shares in the whole labeled file

| Measure | Enthusiast share | n | Shopper share | n | Shopper minus enthusiast | 95% interval |
|---|---:|---:|---:|---:|---:|---|
| charging access | 0.174 | 6402 | 0.034 | 13370 | -0.140 | -0.150 to -0.130 |
| driving range | 0.069 | 6402 | 0.059 | 13370 | -0.010 | -0.018 to -0.003 |
| vehicle price or total cost | 0.176 | 6402 | 0.168 | 13370 | -0.008 | -0.019 to 0.003 |
| purchase incentives or tax credits | 0.032 | 6402 | 0.014 | 13370 | -0.018 | -0.023 to -0.014 |
| battery reliability | 0.118 | 6402 | 0.041 | 13370 | -0.077 | -0.086 to -0.069 |
| dealer or buying friction | 0.031 | 6402 | 0.026 | 13370 | -0.006 | -0.011 to -0.001 |
| environmental impact | 0.065 | 6402 | 0.047 | 13370 | -0.018 | -0.026 to -0.011 |
| owning an electric vehicle | 0.397 | 6402 | 0.634 | 13370 | 0.236 | 0.222 to 0.251 |
| support for electric vehicles | 0.528 | 6402 | 0.818 | 13370 | 0.289 | 0.275 to 0.303 |

### Event windows

Each contrast is the share in the event month or months minus the share in the prior three months, and minus the same months a year earlier. The pre-specified barrier is incentives for August 2022 and for September–October 2025, price for January 2023, and charging for May–June 2023. The other barriers are reported in `framing_event_contrasts.csv`.

| Event | Audience | Baseline | Event share | Baseline share | Difference | 95% interval | Event n | Baseline n |
|---|---|---|---:|---:|---:|---|---:|---:|
| credit_deadline_2025 / purchase incentives or tax credits | enthusiast | prior_3_months | 0.052 | 0.037 | 0.015 | -0.015 to 0.045 | 328 | 436 |
| credit_deadline_2025 / purchase incentives or tax credits | enthusiast | year_ago | 0.052 | 0.014 | 0.038 | 0.002 to 0.074 | 328 | 72 |
| credit_deadline_2025 / purchase incentives or tax credits | shopper | prior_3_months | 0.061 | 0.043 | 0.018 | -0.040 to 0.076 | 98 | 139 |
| credit_deadline_2025 / purchase incentives or tax credits | shopper | year_ago | 0.061 | 0.038 | 0.023 | -0.032 to 0.078 | 98 | 183 |
| ira_2022 / purchase incentives or tax credits | enthusiast | prior_3_months | 0.064 | 0.015 | 0.049 | 0.013 to 0.084 | 236 | 202 |
| ira_2022 / purchase incentives or tax credits | enthusiast | year_ago | 0.064 | 0.000 | 0.064 | 0.032 to 0.095 | 236 | 23 |
| ira_2022 / purchase incentives or tax credits | shopper | prior_3_months | 0.022 | 0.022 | 0.001 | -0.034 to 0.036 | 89 | 277 |
| ira_2022 / purchase incentives or tax credits | shopper | year_ago | 0.022 | 0.000 | 0.022 | -0.008 to 0.053 | 89 | 95 |
| nacs_2023 / charging access | enthusiast | prior_3_months | 0.155 | 0.224 | -0.069 | -0.134 to -0.004 | 554 | 201 |
| nacs_2023 / charging access | enthusiast | year_ago | 0.155 | 0.088 | 0.068 | 0.007 to 0.128 | 554 | 114 |
| nacs_2023 / charging access | shopper | prior_3_months | 0.076 | 0.048 | 0.028 | -0.018 to 0.074 | 184 | 269 |
| nacs_2023 / charging access | shopper | year_ago | 0.076 | 0.024 | 0.052 | 0.008 to 0.097 | 184 | 169 |
| tesla_price_cut_2023 / vehicle price or total cost | enthusiast | prior_3_months | 0.212 | 0.135 | 0.077 | 0.005 to 0.150 | 292 | 148 |
| tesla_price_cut_2023 / vehicle price or total cost | enthusiast | year_ago | 0.212 | 0.052 | 0.161 | 0.087 to 0.234 | 292 | 58 |
| tesla_price_cut_2023 / vehicle price or total cost | shopper | prior_3_months | 0.247 | 0.150 | 0.097 | 0.014 to 0.179 | 146 | 260 |
| tesla_price_cut_2023 / vehicle price or total cost | shopper | year_ago | 0.247 | 0.120 | 0.127 | 0.036 to 0.217 | 146 | 125 |

### Interrupted series

Monthly shares from 12 months before the first event month through 6 months after, weighted by the number of labeled posts. The post coefficient is the level shift at the event month, after a linear pre-trend and a change in slope. Months with fewer than 5 labeled posts are left out. A missing month is skipped, not entered as zero.

| Event | Audience | Level shift | 95% interval | Months | Posts | Post months | Status |
|---|---|---:|---|---:|---:|---:|---|
| credit_deadline_2025 | enthusiast | 0.009 | -0.013 to 0.032 | 17 | 1509 | 7 | ok |
| credit_deadline_2025 | shopper | 0.017 | -0.037 to 0.071 | 16 | 1163 | 4 | ok |
| ira_2022 | enthusiast | 0.039 | 0.013 to 0.066 | 15 | 1298 | 6 | ok |
| ira_2022 | shopper | -0.011 | -0.037 to 0.015 | 19 | 1917 | 7 | ok |
| nacs_2023 | enthusiast | -0.097 | -0.170 to -0.023 | 17 | 2021 | 7 | ok |
| nacs_2023 | shopper | 0.012 | -0.036 to 0.060 | 19 | 1734 | 7 | ok |
| tesla_price_cut_2023 | enthusiast | 0.024 | -0.084 to 0.132 | 17 | 1951 | 7 | ok |
| tesla_price_cut_2023 | shopper | 0.077 | 0.018 to 0.136 | 19 | 1820 | 7 | ok |

Figures: `framing_barrier_shares_monthly.png`, `framing_event_study.png`, `framing_its_level.png`. Event-study points are the share in that relative month minus the share in the month before the event (`framing_event_study.csv`).
<!-- framing-a-end -->
