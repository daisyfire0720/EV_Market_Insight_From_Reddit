# Pilot findings (unvalidated)

Labels in this file are not human-coded. Barrier, stage, and stance shares are model output. Volume ratios are descriptive, not effects.

The paper's question is locked. It asks how the barriers people discuss — price, charging, battery, range, incentives, dealer friction, and environment — shift around the Inflation Reduction Act (August 2022), Tesla's price cuts (January 2023), NACS adoption (May–June 2023), and the end of the federal clean-vehicle credit on 30 September 2025, and whether enthusiast subreddits and shopper subreddits (r/cars, r/Autos, r/carbuying) move together. The hypotheses are in `v2/paper/introduction.md`. Nothing in this file is a test of those hypotheses.

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

Unvalidated as a statement about barriers. This is a count of posts the extractor kept, not a coded result. On 17 October 2026 the per-subreddit submission files (infohash `3e3f64dee22dc304cdd2546254ca1f8e8ae542b4`) were streamed in full: 1,077,267 lines, 0 bad JSON lines. The word-boundary filter kept **13,976** posts dated 2015-01-01 through 2025-12-28: r/cars 12,210, r/Autos 1,428, r/carbuying 338. The files end in 2025, so they do not contain the 485 API posts from January–September 2026. Against the API aggregate of 16,993 (2015 through September 2026), the dump total is lower in every subreddit, and the gap is wider in later years of r/cars (102 posts in 2015, 328 in 2022). The API search matches tokens the word-boundary rule skips, including `ioniq` inside `ioniq5`. The two columns were not forced together. Year table: `v2/data_card/dump_vs_api_counts.csv`. The NLI labels further down were not rerun on these 13,976 posts.

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
