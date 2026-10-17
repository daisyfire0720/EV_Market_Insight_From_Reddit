# How purchase barriers move when EV policy and prices move

Working title. Transportation Research Part D shape. This is the introduction and the research questions for Framing A, which is the paper. The skeleton around it is `outline.md`. Numbers below are the ones already computed in `v2/data_card/` and `v2/outputs/`. They are not targets. Barrier shares from the pilot are not evidence for the hypotheses.

## Introduction

Electric-vehicle policy treats price, charging, range, and information as the reasons households hold back. Those claims are usually tested with surveys of people who already say they might buy. A survey misses the wording people use when nobody has asked them, and it misses the particular month in which a credit, a price cut, or a charging-standard announcement lands.

Reddit is a public record of that wording. Ruan and Lv (2022) showed that the record is large enough to recover topics and a sentiment trend: 3,437,917 posts from January 2011 through December 2020, a 20-topic LDA model, and VADER. Ruan and Lv (2023) showed that the Reddit pattern is not the same pattern as Twitter over that same decade. Ashby, Weir, and Fussey (2025) showed that a topic model plus a language model can organize charging talk, on just under 100,000 tweets, with manual review of the themes. Yu, Que, Cushing, Pierce, Shen, Kejriwal, Yao, and Zhu (2025) showed what a check on document-level labels looks like, on charging-station reviews rather than on Reddit: two annotators, Cohen's κ of 0.81 for sentiment on 3,188 reviews, human inter-rater κ of 0.71 for problem categories, and language-model accuracy of 0.77 to 0.82 on those categories.

None of those papers measures, at the post and with that kind of check, the barrier a writer is raising, the stage of the purchase they are in, and whether they are for or against the technology, through the dates that now organize US electric-vehicle buying. Those dates are the Inflation Reduction Act (Pub. L. 117-169, 16 August 2022), Tesla's US price cuts reported on 13 January 2023, Ford's charging agreement of 25 May 2023 and GM's NACS announcement of 8 June 2023, and the end of the federal clean-vehicle credit for vehicles acquired after 30 September 2025 (Pub. L. 119-21, 4 July 2025; 26 U.S.C. § 30D). The Ruan corpora stop in December 2020.

The communities are part of the question. Posts in r/electricvehicles, r/teslamotors, r/BoltEV, r/Rivian, r/Ioniq5, r/KiaEV6, and r/evcharging are the enthusiast record. Posts in r/cars, r/Autos, and r/carbuying that actually mention an electric vehicle are the closest public record this project has of mainstream car shoppers. A subreddit is not a person. Owners post in r/cars, and shoppers post in r/teslamotors. Journey stage is therefore a label on the post, not a property of the subreddit name.

The instrument is a zero-shot language model, because the corpus is too large to code by hand and because the earlier papers' tools do not emit a barrier. VADER emits a sentiment score. LDA emits a topic. A keyword list emits a stage, and the list in the existing repository matches `ev` inside `every`. Gilardi, Alizadeh, and Kubli (2023) are the reason to try a language model as an annotator, and the reason not to stop there: the model has to be checked against people. Two annotators, a frozen codebook, and Cohen's kappa per barrier, stage, and stance are the methods core. Ha, Marchetto, Dharur, and Asensio (2021) are the template for a supervised classifier once those human labels exist. They are not a result in this draft.

The pilot already has a monthly submission census for the seven enthusiast subreddits. Complete months from January 2015 through September 2026 sum to 603,435 submissions. It also has an API count of 16,993 keyword submissions in the three shopper subreddits over the same span. A dump extract on 17 October 2026 kept 13,976 of those shopper submissions from 2015 through 2025. The files stop before 2026, and the local keyword rule is stricter than the API search, so 13,976 is not a correction of 16,993. Those are counts of how many posts exist. They are not a finding about which barrier people raised. The label shares in `v2/outputs/FINDINGS.md` come from an earlier, incomplete text file and an unchecked model. They were not recomputed on the 13,976 dump posts. They are not used below as support for any hypothesis.

## Research questions

1. Which barriers — price, charging, battery, range, incentives, dealer friction, and environment — are common in enthusiast subreddits and in shopper subreddits from 2015 through September 2026, and which become more or less common?
2. Around each shock listed above, does the share of posts about incentives, price, or charging differ from the prior three months and from the same months a year earlier?
3. Do shopper subreddits place more of their barrier talk on price and on dealer friction, and less of their talk in the ownership stage, than enthusiast subreddits?

Submission volume is reported next to those shares. A change in how many posts were made is not a change in what the posts were about.

## Hypotheses

These are expectations to test after the human labels exist. The pilot does not test them. Each comparison is the event window against the prior three months and against the same months one year earlier. Neither comparison is an effect of the policy.

**H1. Incentives.** The share of posts coded `incentives_policy` is higher in August 2022, in July 2025, and in September 2025 than in both comparison windows. August 2022 is the month the Inflation Reduction Act was enacted. July 2025 is the month of the statute that ends the credit. September 2025 is the acquisition deadline of 30 September 2025.

**H2. Price.** The share of posts coded `price_tco` is higher in January 2023, the month of the reported Tesla price cuts, than in both comparison windows.

**H3. Charging.** The share of posts coded `charging_access` is higher in May and June 2023, the months of the Ford and GM connector announcements, than in both comparison windows. The codebook does not have a separate code for the connector standard. Connector talk is scored as charging access when the post is about getting a charge.

**H4. Shoppers and enthusiasts.** Among posts that receive at least one barrier, the shopper subreddits have a higher share of `price_tco` and of `dealer_friction` than the enthusiast subreddits. They also have a lower share of the ownership stage. The stage label, not the subreddit name, is what makes a post "ownership." H4 can fail even if the subreddit totals differ, if the stage labels say the shopper subreddits are full of owners.

**H5. The instrument.** On the double-coded sample, Cohen's kappa between the two annotators, and the agreement between the zero-shot labels and the adjudicated labels, are high enough to support H1–H4. The rule already fixed in the design is the operational one: if the lower bound of the confidence interval on barrier kappa is below 0.60, add 200 adjudicated posts before a substantive claim, and if it is still too low, hold the Transportation Research Part D claim. VADER is compared on stance only. It is not a barrier classifier. The substring stage rule is reported for its false matches, not as a measure of the market.

## What this draft does not claim

No human has coded the corpus. Nothing in the pilot is a statement about what "the public" thinks. A higher share in an event month is not an effect of the statute or of the price cut. English-language Reddit is not the US household population. The unit in this design is the submission. Ruan and Lv's mass is mostly comments, so a comparison of levels with those papers is not like-for-like.

## Sources for the dates and the prior studies

Statute and news sources for the five windows are in section 6 of `v2/research_design.md`. The papers named above are the reference list in `outline.md`. No citation was added in this draft beyond that list.
