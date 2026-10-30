# Alternative framing, not a decision

Daisy has not chosen this. The working title and the abstract in `manuscript.md` are unchanged. This file is a draft she can adopt or ignore.

## Alternative title

Stage before shock: enthusiast and shopper funnels in electric-vehicle talk on Reddit, 2015–2026.

## Alternative abstract

Surveys of people who say they might buy an electric vehicle record price, range, and charging as barriers. They miss the wording used when nobody is asking. This paper reads English-language Reddit submissions from 2015 through September 2026 and asks where the writer sits in a six-stage purchase funnel, and which topic the post carries. Enthusiast posts are a week sample of seven electric-vehicle subreddits. Shopper posts are keyword-matched submissions from r/cars, r/Autos, and r/carbuying. Labels are from Qwen2.5-14B-Instruct, run locally, with a smaller zero-shot model beside them. Two annotators are to code 800 posts. That sheet is still blank, so the labels are unvalidated.

The descriptive split is the result the file can support. Of 9,036 enthusiast submissions with a codebook stage, 26.8% are onboarding, the early period of ownership. Of 13,743 shopper submissions, 32.7% are awareness. Purchase is 2.8% and 0.9%. Long-term ownership is 1.9% and 2.0%. The stage-by-family grid is not the same mix at those two stages. Jumps at four US dates were fit as a regression discontinuity in time and are reported as a sensitivity. Enthusiast incentive talk at 1 January 2023 is +3.5 points, interval −5.3 to +7.2 (small model +5.0). Shopper price talk at the 13 January 2023 Tesla cuts is +10.5, interval −2.2 to +24.2 (small model +13.1). Both intervals include zero. Dropping posts with a score under 2 cuts them to +0.5 and +0.8. A balanced comment sample does not reproduce them (−5.6 and −2.9 on the comments drawn for those cutoffs). The jumps are consistent with a change in what was posted. They are not effects of the statute or the price cut.

## Contributions

The paper's contribution, under this framing, is a document-level description of where electric-vehicle talk sits: enthusiast submissions are concentrated at onboarding (26.8% of 9,036) and shopper submissions at awareness (32.7% of 13,743), with a stage-by-family grid that keeps charging, price, incentives, and the other families from being collapsed into one ownership class. The local 14B labels and the small-model baseline are the measurement, still unvalidated, with stage kappa about 0.11 between the two models and an empty cell for the human kappa. The regression discontinuity in time, the score filter, the fake-date checks, and the balanced comment sample are exploratory sensitivities. They show that the two jumps the small model also pointed to are not distinguishable from zero and are not repeated in the comment draw.

## What would change in sections 1, 7, and 8

Section 1 would lead with research question 1, the stage-by-family comparison of the enthusiast file and the shopper file. Question 2, the four cutoff jumps, would be labeled exploratory and would follow the descriptive question. The sentence that a jump is not an effect would stay. The hypotheses file would not be rewritten until she asks.

Section 7 would stay the small-model baseline, with the same unvalidated warning. It would be introduced as the baseline for the funnel shares and for the exploratory jumps, not as evidence that a shock moved the barrier mix. Its discontinuity table would be cross-referenced to the sensitivity in section 8 rather than read as the paper's result.

Section 8 would open on the six-stage shares and the stage-by-family heatmap (current sections 8.2 and 8.3). The discontinuity (8.4), the transitions (8.5), and the comment sample (8.7) would sit under one heading, exploratory sensitivity, with the same numbers, the same "consistent with" language, the same three fake-date demotions, the same score-filter fragility, and the same empty donuts on the short January sides. The kappa table and the human-kappa placeholder (8.6) would stay with the measurement, not with the shocks. No number in the current tables would be replaced.
