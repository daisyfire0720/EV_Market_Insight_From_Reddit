# Codebook for the Framing A annotation sheet

Codes apply to the submission as a whole (title plus the excerpt). A post may receive more than one barrier. Stage is one code. Stance is one code. The illustrations were written for this codebook. They are not quotations from the corpus.

The sheet is `annotation_sheet.csv`. Leave a cell blank only when the excerpt is missing or unreadable, and say so in `notes`. Otherwise every barrier cell is `1` or `0`, `stage` is one of the six stage codes, `onboarding` is `yes`, `no`, or `na`, and `stance` is one of the four stance codes. `none` is `1` only when every other barrier is `0`.

Do not look up the model key while coding. `annotation_model_key.csv` is for the scoring script after both coders finish. `annotation_preview.csv` is the 25-row practice file and is not part of the 800.

The subreddit is context for the writer's situation. It is not itself a stage. A post in r/teslamotors can be a shopper, and a post in r/cars can be an owner.

## How the 800 were drawn

The validation plan already fixed 800 submissions so two people can finish the set (the range discussed for this paper was 500 to 1,000; Yu et al. used 3,188 reviews for a binary sentiment task). A plain year-band sample would contain only a handful of incentive and dealer posts, and kappa on those labels would rest on those handfuls.

The draw therefore fills, in order:

1. Up to 40 posts per audience with each barrier turned on by the model, rare labels first (incentives, then dealer friction). Forty positives is the floor that keeps a binary kappa from being a story about five posts. Where the corpus has fewer than 40, the draw takes all of them.
2. Up to 25 posts per audience inside each of the four event windows (August 2022, January 2023, May–June 2023, September–October 2025).
3. Up to 50 posts per audience in each year band (2015–2017, 2018–2020, 2021–2023, 2024–latest), so the sheet is not only rare-label posts.
4. If those layers pass 800, posts that were added only as year-band or leftover fill are dropped first. Posts added for a barrier floor are kept.

The practice ids are excluded. The achieved counts are in `v2/outputs/annotation_sample_summary.csv` after the draw script runs. Coders should ignore the model when the text disagrees with it. The model was used only to make sure rare labels appear.

## Barriers

Mark `1` when the post is about that barrier, even if the writer raises it to dismiss it. Mark `0` when it is absent. More than one `1` is expected.

| Code | Apply when | Do not apply when |
|---|---|---|
| `charging_access` | Getting a charge: home electrical work, apartments, public stations, queues, broken or blocked chargers, connector type, where to plug in | The only energy mention is gasoline prices, or the charger is only the location of a dealer visit with no charging problem |
| `range` | Whether the car can cover a trip, winter loss, range anxiety, miles per charge on a journey | Battery aging or replacement with no claim about distance |
| `price_tco` | Purchase price, lease payment, insurance, depreciation, electricity versus fuel, repair bills as a cost | A tax credit or rebate named as a policy, which is `incentives_policy`. Mark both if the post discusses the price and the credit |
| `incentives_policy` | Tax credit, rebate, eligibility, the IRA, a credit deadline, a government mandate or ban | A private discount or a dealer’s own “market adjustment,” with no public incentive |
| `battery_reliability` | Degradation, replacement, recall, fire, cold-weather durability, “will the battery last” | Range on a single trip with no durability claim |
| `dealer_friction` | Markup, added fees that are a buying complaint, inventory, waitlist, pressure, bait-and-switch, the buying transaction at a dealer | Ordinary service of a car the writer already owns, or a charger that merely sits at a dealer |
| `environment` | Emissions, mining, the grid mix, whether the car is actually cleaner | A color, a brand, or “green” used only as a paint name |
| `none` | No barrier above. If any other barrier is `1`, `none` is `0` | — |

Decision rules that come up often:

- A dollar figure is not automatically `price_tco`. “The credit took $7,500 off” is `incentives_policy`. “It was $7,500 cheaper than the gas car, before any credit” is `price_tco`. Both sentences in one post get both codes.
- One broken public charger is `charging_access`. It is not `environment` and it is not, by itself, opposition.
- “The dealer wants $5,000 over sticker” is `dealer_friction` and usually `price_tco`. “I dropped it at the dealer for a tire” is neither, unless the post is about the price of the repair (`price_tco`) or a battery fault (`battery_reliability`).
- Connector talk (NACS, CCS, adapter) is `charging_access` when the writer is talking about being able to charge. A pure stock-price comment about the companies is not.

Worked examples (invented):

- “Third floor, no garage, and the only public station was blocked again.” `charging_access=1`. The others `0`. `none=0`.
- “We drove to Denver and had to stop twice. I don’t want to do that with kids.” `range=1`.
- “Insurance went up $80 a month and the lease is still $612.” `price_tco=1`.
- “The point-of-sale credit is gone after September, so the out-the-door price jumps.” `incentives_policy=1` and `price_tco=1`.
- “The pack warranty is 8 years and I’m at year 9. Capacity is down and they want a new pack.” `battery_reliability=1`.
- “Two dealers, both with a $4,000 market adjustment and a doc fee I can’t decline.” `dealer_friction=1` and `price_tco=1`.
- “The mining footprint is the part I can’t get past, whatever the tailpipe says.” `environment=1`.
- “Here is a photo of the wrap I put on it.” Every barrier `0`, `none=1`.

## Stage

One code. Code the writer’s situation, not the topic they mention and not the subreddit.

| Code | The writer is |
|---|---|
| `awareness` | Learning what an EV is, or whether to consider one at all |
| `consideration` | Comparing models, brands, or an EV versus a gasoline car, without a near-term purchase |
| `evaluation` | Working a budget, a credit, a commute, or a charger for a purchase they are close to making |
| `purchase` | At a dealer, placing an order, signing, or taking delivery |
| `ownership` | Already has the vehicle: living with it, service, road trips in it, a problem with the car they drive |
| `unclear` | The stage cannot be told, or the post is not about a personal vehicle decision (news, policy in the abstract, a joke, a photo with no decision) |

`onboarding` is an optional sub-code. It is meaningful only when `stage` is `ownership`.

| `onboarding` | Use when |
|---|---|
| `yes` | The early period with the car: delivery week, first charge, learning the controls, the first months |
| `no` | Later ownership: a car they already know, a repair years in, a trip in a familiar car |
| `na` | `stage` is anything other than `ownership` |

Daisy has not chosen between one ownership stage and Ryan’s onboarding split. The taxonomy PDF is not in the repo. Coders always fill both columns. The scoring script can collapse them either way:

- One ownership stage: ignore `onboarding`. `ownership` stays `ownership`.
- Split: `ownership` plus `onboarding=yes` is onboarding; `ownership` plus `onboarding=no` is long-term ownership. If `onboarding` is blank on an ownership post, the collapsed code stays `ownership` until the two coders resolve it.

Worked examples (invented):

- “What even is the difference between a hybrid and this?” `awareness`. `onboarding=na`.
- “Torn between the Ioniq and a used Model 3, not buying this year.” `consideration`. `onboarding=na`.
- “My commute is 46 miles and the credit would make the payment fit. Can I charge in a rental?” `evaluation`. `onboarding=na`.
- “Picking it up Saturday. They still have the markup on the contract.” `purchase`. `onboarding=na`.
- “Took delivery Tuesday and I can’t get the scheduled charge to stick.” `ownership`, `onboarding=yes`.
- “Three years in, the control-arm noise is back.” `ownership`, `onboarding=no`.
- “Reuters says the credit ends in September.” `unclear` if the writer does not place themselves in a purchase. `onboarding=na`.

## Stance

One code. Stance is about electric vehicles as something to buy, keep, or support. A complaint about one product incident is not automatically opposition.

| Code | Meaning |
|---|---|
| `support` | The writer recommends EVs, defends them, or describes a good outcome they attribute to the technology |
| `oppose` | The writer argues against buying or subsidizing EVs, beyond a single incident |
| `mixed` | Both, or a complaint that leaves the writer’s position unresolved |
| `no_stance` | A factual or procedural post, or a brand complaint that does not generalize to EVs |

Worked examples (invented):

- “Best car I’ve owned. I would not go back.” `support`.
- “Every one of these is a coal car and the credit is a gift to the manufacturers.” `oppose`. Also `environment=1` and `incentives_policy=1`. Stage `unclear` unless they are refusing a purchase of their own.
- “I love the car and I hate that the only station in town is down.” `mixed`.
- “The service appointment is Thursday at 8.” `no_stance`.
- “The stall was offline this morning.” `no_stance` or `mixed` only if they go on to a judgment about EVs. The broken stall alone is not `oppose`.

## Two coders

Each person fills a copy of the sheet, puts their name in `annotator`, and does not edit the other file. Disagreements are counted by `v2/score_agreement.py`. The script does not pick a winner. Adjudication, if Daisy wants it, is a later meeting. Items still unresolved stay out of the gold set.

The script reports Cohen’s kappa and Krippendorff’s alpha per field, precision/recall/F1 of the model against the items where the two coders agree, and a Rogan-Gladen correction of the model’s positive rate. A blank on one sheet and a code on the other is a missing-value mismatch, not a third category.
