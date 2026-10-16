# Pilot findings, 16 October 2026

Numbers below are copied from `v2/data_card/` and the other files in `v2/outputs/`. Volume is a census of submissions. Text results are a designed sample. None of the labels have been checked by a human.

## Data

Arctic Shift returned a count for every core subreddit-month from January 2015 through October 2026. October 2026 is partial (the pull was on 16 October): 2,440 submissions. Complete months, January 2015 through September 2026, sum to **603,435** submissions.

| Year | Submissions |
|---|---:|
| 2015 | 9,074 |
| 2016 | 19,902 |
| 2017 | 24,753 |
| 2018 | 40,089 |
| 2019 | 60,455 |
| 2020 | 46,208 |
| 2021 | 62,926 |
| 2022 | 83,904 |
| 2023 | 73,127 |
| 2024 | 75,232 |
| 2025 | 65,916 |
| 2026, Jan–Sep | 41,849 |

January–September was 57,104 in 2024 and 52,616 in 2025, against 41,849 in 2026. r/teslamotors was 82.3% of the seven-subreddit total in 2015 and 7.8% in January–September 2026. The later total is carried by r/Rivian, r/Ioniq5, r/KiaEV6, and r/electricvehicles. 2020 (46,208) is below 2019 (60,455). Inside 2020, January was 5,185 and April was 2,579.

The text file has 7,214 deduplicated submissions: 1,440 r/teslamotors, 1,415 r/electricvehicles, 1,125 r/BoltEV, 877 r/Rivian, 750 r/Ioniq5, 714 r/KiaEV6, 623 r/evcharging, 228 r/cars, 33 r/Autos, 9 r/carbuying. It is one deterministic week per quarter, plus a fixed week around each coded event, not a simple random sample. Cleaning kept 6,656 (22 bot rows removed, 536 under five tokens).

## Event windows

Mean monthly submissions in the seven core subreddits. Ratios are descriptive.

| Window | Mean | Versus prior 3 months | Versus a year earlier |
|---|---:|---:|---:|
| August 2022 | 8,080 | 1.076 | 1.570 |
| January 2023 | 6,154 | 0.987 | 0.929 |
| May–June 2023 | 5,930.5 | 1.032 | 0.808 |
| July 2025 | 5,567 | 0.991 | 0.903 |
| September–October 2025 | 4,976.5 | 0.908 | 0.825 |

August 2022 is close to the three months before it. The comparison with August 2021 is large because volume was still rising, not because the table isolates the Act. The price-cut month, the NACS months, the July 2025 enactment month, and the September–October 2025 deadline are flat or lower than both comparisons. In these communities, the end of the federal credit does not appear as a surge in how many posts were made.

## Keyword stage rule

Substring matching assigned 43.1% of the 6,656 posts to Consideration. Word-boundary matching assigned 25.0% to Consideration and 50.5% to Unclear. That gap is the bug in the existing funnel.

## Topics and sentiment

LDA (8 topics, seed 42) gives every document a probability. The mean of the maximum is 0.629. The largest assigned topics are a Bolt / used-car cluster (1,207 documents) and a service / app / issue cluster (1,165). Mean VADER compound is 0.196 in brand communities and 0.204 in r/electricvehicles plus r/evcharging, and 0.089 in the 254 mainstream posts that survived cleaning.

## Zero-shot labels

On 400 stratified posts, `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` (revision `262ae02f29173eec1c250f90804dc7edc677dcff`), threshold 0.5: charging access 20.5%, price or total cost 18.0%, battery reliability 11.5%, environment 8.3%, range 7.0%, incentives 4.0%, dealer friction 4.0%. Stage is mostly owning (43.3%) or unclear (27.3%). Stance is support 57.0%, mixed 26.3%, no stance 16.0%, opposition 0.75% (3 posts).

Before the corpus run, five invented sentences were used to choose the hypothesis template. The model caught charging, range, an environmental objection, and a tax credit. It called a dealer-markup sentence a price post. The dealer share may be low because of that. No human has labeled the 400.

## What this does not show

The mainstream keyword sample is too small, and too full of timeouts (56 on the second pass), to compare shoppers with owners. Comments were probed, not counted. Event ratios are not effects. The NLI shares are a first look at a small CPU model.
