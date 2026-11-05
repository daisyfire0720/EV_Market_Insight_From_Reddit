# Codebook for the 800-post sheet

Coder A is Daisy. Coder B is Ryan (GitHub Ryan-yu-wei). Code independently. The taxonomy is Daisy's reference note, committed at `v2/docs/reference_taxonomy_6stage_8family.pdf`. Code the rows where `calibration` is `yes` first (50 posts). Compare those, freeze any wording changes in this file, then code the other 750. The 50 calibration posts are left out of kappa. Ryan's short instructions are `v2/annotation/README_FOR_RYAN.md`.

Do not open `annotation_model_key.csv`. It is the small model's predictions, for the scoring script only. The code cells on `annotation_sheet.csv` are empty on purpose.

No one has filled this sheet. Do not copy labels from a model into it.

## What a row is

One Reddit submission. `title` and `text_excerpt` are the text. The excerpt is at most 500 characters. Usernames that looked like `u/name` are replaced with `u/[redacted]`.

Code the writer's position, not the subreddit's reputation. A post in r/teslamotors can be a shopper. A post in r/cars can be an owner.

## Columns you fill

Use `1` or `0` for every family and every aspect. Use one stage code and one stance code. Blank means you skipped the item. A blank is missing, not a third category.

| Column | What to enter |
|---|---|
| Eight family columns | `1` if that family is a real subject of the post, otherwise `0`. More than one family can be `1`. |
| Seven aspect columns, and `none` | `1` if that finer barrier is present. `none` is `1` only when every aspect is `0`. |
| `stage` | Exactly one of the six stage codes, or `unclear`. |
| `stance` | Exactly one of `support`, `oppose`, `mixed`, `no_stance`. |
| `annotator` | Daisy writes `Daisy`. Ryan writes `Ryan`. Use that same string on every row you code. |
| `notes` | Optional. Use it when the stage was a close call. |

`calibration` is already filled. Do not change it.

## Stage: where the writer is

Assign the stage from the writer's place in the journey, not from the topic word alone. Charging can be Evaluation if they are still deciding, or Onboarding if they already bought the car.

| Code | Name | Definition | Typical signals, not rules |
|---|---|---|---|
| `awareness` | Awareness / Need Formation | Earliest discovery. They are asking whether an EV is worth considering and what problem it might solve. | General curiosity, a first question, EV versus a regular car at a high level. |
| `consideration` | Consideration / Shortlisting | They are narrowing brands, models, classes, or broad alternatives. | Tesla vs Ioniq 5, best EV SUV, used Bolt vs Leaf, hybrid vs EV as a shortlist. |
| `evaluation` | Evaluation / Practical Fit | They are testing whether an EV works for their life, on money or on daily use. | Apartment charging, range fit, winter, insurance, incentives, total cost, before a purchase. |
| `purchase` | Purchase / Transaction | The discussion has moved from deciding to carrying out the purchase. | Markup, financing, lease terms, the order, inventory, delivery timing. |
| `onboarding` | Onboarding / Early Ownership | They just bought it and are learning the car and the ecosystem. | First charging setup, app confusion, first-week impressions, the first service question. |
| `long_term_ownership` | Long-Term Ownership / Retention | Later ownership: reliability, maintenance, satisfaction, replacement. | Degradation, warranty, service centers, resale, regret or satisfaction after real use. |
| `unclear` | Unclear | The post does not show where they are. A news link with no journey is `unclear`. | |

Onboarding and long-term ownership are separate. Do not collapse them into one ownership code.

### Worked stage examples from the taxonomy note

These are illustrations, not posts from the corpus.

| Example | Stage | Family |
|---|---|---|
| Tesla Model Y vs Hyundai Ioniq 5 for a family car | `consideration` | `purchase_selection` |
| Can I make an EV work if I live in an apartment and street park? | `evaluation` | `charging_energy` |
| I just bought a Bolt — how should I set up home charging and the app? | `onboarding` | `charging_energy` |
| My EV insurance is much higher than expected — is this normal? | `evaluation` | `cost_incentives` |
| Dealer added a markup and the delivery keeps slipping | `purchase` | `dealer_access` |
| After two winters, my battery range seems noticeably worse | `long_term_ownership` | `battery_range` |

A further illustration, also invented: "Are electric cars even worth looking at?" is `awareness`. "Every EV is a coal car" has no journey and is `unclear`, stance `oppose`, aspect `environment`.

## Families: what the post is about

| Code | Name | Scope |
|---|---|---|
| `purchase_selection` | EV Purchase and Vehicle Selection | Brand or model choice, body class, new versus used, a shortlist. |
| `charging_energy` | Charging and Energy Access | Home charging, public charging, apartments, speed, compatibility. |
| `battery_range` | Battery, Range, and Real-World Performance | Degradation, range, winter range, road-trip efficiency. |
| `cost_incentives` | Cost, Incentives, and Financial Feasibility | Price, payment, insurance, tax credits, lease, total cost. |
| `dealer_access` | Dealer, Purchase Process, and Market Access | Markup, inventory, wait, order flow, delivery. |
| `reliability_service` | Reliability, Service, and Ownership Experience | Repair, service centers, warranty, software trouble after purchase. |
| `policy_market` | Policy, Regulation, and Market Environment | Credit rules, emissions rules, charging policy, the broad market. |
| `transition_narratives` | ICE / Hybrid / EV Transition Narratives | Switching from gasoline, hybrid versus EV, skepticism about the transition itself. |

A make or model still belongs to one of these families. Do not invent a ninth family for Tesla or Rivian. More than one family may be present. If the post is only a joke with no subject above, leave every family `0`.

## Aspects: the finer barriers

These sit under the families. Mark each one that the post is actually about.

| Code | Mark `1` when the post is about |
|---|---|
| `charging_access` | Getting a charge: home, public, apartment, a broken or blocked stall, connector access. |
| `range` | How far it goes, including winter and highway range. |
| `price_tco` | Vehicle price, payment, insurance, or total cost, apart from a named incentive. |
| `incentives_policy` | A tax credit, rebate, or eligibility rule. |
| `battery_reliability` | Battery health, degradation, or a battery failure. |
| `dealer_friction` | Markup, add-on fees, inventory games, or a dealer who blocks the sale. Ordinary service at a dealer is `reliability_service` / not this aspect, unless the problem is the purchase itself. |
| `environment` | Emissions, the grid, or whether the car is cleaner. |
| `none` | `1` only if all seven aspects are `0`. |

A post can be `charging_energy` as a family and `charging_access` as an aspect. The family is the umbrella. The aspect is the specific barrier.

## Stance

| Code | Use it when |
|---|---|
| `support` | They are for electric vehicles as a class, or they are buying or keeping one without a general rejection. |
| `oppose` | The claim reaches beyond one incident: EVs as a class are a bad idea. |
| `mixed` | Both, or a complaint that leaves the general position unresolved. |
| `no_stance` | A factual or procedural post, or a brand complaint that does not generalize. |

One broken charger is not `oppose` by itself.

## How the 800 were chosen

The draw is stratified so the sheet is not only the common case (owners talking about price). Inside each audience (enthusiast subreddits, and r/cars, r/Autos, r/carbuying):

- up to 20 posts in each sampling stage the current file can see,
- up to 15 posts in each topic family the current file can see,
- rare aspects first (incentives, dealer friction),
- then posts inside the four event windows.

The sampling stage is not a code. The small model has one ownership class. A short lexicon was used only to put some of those ownership posts into the onboarding pool and some into the long-term pool, so both stages show up on the sheet. If the lexicon did not fire, the post was not forced into either stage. You still assign the stage yourself. The practice file `annotation_preview.csv` is excluded.

`sample_manifest.csv` lists id, layer, audience, subreddit, month, year band, and whether the row is in the calibration set. It does not list the model's stage or family.

## After both sheets exist

From `v2/`:

```
python score_agreement.py \
  --coder-a annotation/coder_daisy.csv \
  --coder-b annotation/coder_ryan.csv \
  --model-key annotation/annotation_model_key.csv \
  --corpus outputs/framing_labels.csv \
  --out outputs/agreement
```

The script computes Cohen's kappa and Krippendorff's alpha per field, the model's precision, recall, and F1 against the items you both coded the same way, and a Rogan-Gladen prevalence correction. It drops the calibration rows from the main table and writes them separately as `calibration_agreement.csv`. It does not invent a code. Disagreements are counted and left out of the gold set until you adjudicate them.

If the lower bound of the confidence interval on the aspect kappas is below 0.60, add 200 adjudicated posts before a substantive claim.
