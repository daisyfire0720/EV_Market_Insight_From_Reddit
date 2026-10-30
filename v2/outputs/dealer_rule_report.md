# Dealer-friction rule on hand-written sentences

These sentences were written for the codebook. They are not Reddit posts. Model `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` revision `262ae02f29173eec1c250f90804dc7edc677dcff`. Threshold 0.5.

Before: the original label `dealer or buying friction` is on when its score is at least 0.5.
After: that score, or the dealer-friction lexicon. Ordinary service and charging-at-a-dealer are forced off when the lexicon does not match.
Alternate label tried and not adopted: `dealer markup or dealer pressure during the purchase`, same template `This text is about {}.`.

Sentences: 18. Before matched the intended label on 10. After matched on 18. The alternate label matched on 11.

Intended dealer sentences: 9. Before on: 3. After on: 9.
Intended non-dealer sentences: 9. Before false positives: 2. After false positives: 0.

| Id | Intended | Before | After | Reason | Dealer score | Price score | Alternate on |
|---|---|---|---|---|---:|---:|---|
| markup_market_adjustment | yes | False | True | lexicon | 0.094 | 0.822 | False |
| markup_made_price | yes | True | True | nli+lexicon | 0.873 | 0.589 | True |
| no_inventory | yes | False | True | lexicon | 0.360 | 0.001 | False |
| doc_fee_addendum | yes | False | True | lexicon | 0.142 | 0.293 | True |
| refused_msrp | yes | False | True | lexicon | 0.421 | 0.988 | False |
| market_adjusted_firm | yes | False | True | lexicon | 0.147 | 0.154 | False |
| would_not_drop | yes | True | True | nli+lexicon | 0.716 | 0.025 | False |
| pure_price | no | False | False | off | 0.015 | 0.979 | False |
| service_visit | no | True | False | suppressed_service_or_charging_location | 0.887 | 0.022 | False |
| charging_at_dealer | no | True | False | suppressed_service_or_charging_location | 0.542 | 0.217 | False |
| tax_credit_price | no | False | False | off | 0.009 | 0.917 | False |
| apartment_charger | no | False | False | off | 0.008 | 0.002 | False |
| coal_car | no | False | False | off | 0.008 | 0.155 | False |
| fair_quote | no | False | False | off | 0.131 | 0.768 | False |
| finance_pressure | yes | True | True | nli | 0.802 | 0.346 | False |
| winter_range | no | False | False | off | 0.001 | 0.001 | False |
| positive_price | no | False | False | off | 0.027 | 0.537 | False |
| waitlist_only | yes | False | True | lexicon | 0.418 | 0.004 | False |
