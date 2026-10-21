"""Draft manuscript for Transportation Research Part D.

Every number is read from ``v2/outputs``. The draft says the labels are
unvalidated. Citations are only the ones already resolved in
``research_design.md``. Anything else is marked [CITATION NEEDED].
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ev_v2.round6_report import _fmt, moved_rows


def _share_table(overall: pd.DataFrame) -> list[str]:
    lines = [
        "| Measure | Enthusiast share | Shopper share |",
        "|---|---:|---:|",
    ]
    measures = list(dict.fromkeys(overall["measure"].tolist()))
    for measure in measures:
        enth = overall[(overall["measure"] == measure) & (overall["audience"] == "enthusiast")]
        shop = overall[(overall["measure"] == measure) & (overall["audience"] == "shopper")]
        if enth.empty or shop.empty:
            continue
        lines.append(
            f"| {measure} | {_fmt(enth.iloc[0]['share'])} (n={int(enth.iloc[0]['n'])}) | "
            f"{_fmt(shop.iloc[0]['share'])} (n={int(shop.iloc[0]['n'])}) |"
        )
    return lines


def _contrast_table(boot: pd.DataFrame) -> list[str]:
    rows = boot[(boot["primary"] == True) & (boot["reported"] == True)].copy()  # noqa: E712
    lines = [
        "| Event | Audience | Baseline | Difference | Bootstrap interval | Baseline posts | Includes 0 | Thin month |",
        "|---|---|---|---:|---|---:|---|---|",
    ]
    for _, row in rows.iterrows():
        includes = "no" if bool(row["boot_excludes_zero"]) else "yes"
        thin = "yes" if bool(row["thin_window"]) else "no"
        lines.append(
            f"| {row['event_id']} | {row['audience']} | {row['baseline']} | {_fmt(row['diff'])} | "
            f"{_fmt(row['boot_lo'])} to {_fmt(row['boot_hi'])} | {int(row['n_baseline'])} | {includes} | {thin} |"
        )
    return lines


def _its_table(its: pd.DataFrame) -> list[str]:
    rows = its[its["primary"] == True].copy()  # noqa: E712
    lines = [
        "| Event | Audience | Level shift | Month-bootstrap interval | Months | Includes 0 |",
        "|---|---|---:|---|---:|---|",
    ]
    for _, row in rows.iterrows():
        includes = "no" if bool(row["boot_excludes_zero"]) else "yes"
        lines.append(
            f"| {row['event_id']} | {row['audience']} | {_fmt(row['post'])} | "
            f"{_fmt(row['boot_lo'])} to {_fmt(row['boot_hi'])} | {int(row['n_months'])} | {includes} |"
        )
    return lines


def _fill_change_sentences(out_dir: Path) -> list[str]:
    before = out_dir / "framing_block_bootstrap_before_fill.csv"
    after = out_dir / "framing_block_bootstrap.csv"
    its_before = out_dir / "framing_its_bootstrap_before_fill.csv"
    its_after = out_dir / "framing_its_bootstrap.csv"
    if not (before.exists() and after.exists()):
        return ["The thin-baseline comparison has not been written yet."]
    lines = ["Filling the four thin enthusiast months changed these zero-inclusion decisions (design-aware bootstrap, unvalidated):"]
    lines.append("")
    moved = moved_rows(pd.read_csv(before), pd.read_csv(after))
    if moved.empty:
        lines.append("- No primary event-window interval changed whether it includes zero.")
    for _, row in moved.iterrows():
        before_word = "excluded" if row["excluded_before"] else "included"
        after_word = "excludes" if row["excluded_after"] else "includes"
        lines.append(
            f"- {row['event_id']}, {row['audience']}, {row['baseline']}: {_fmt(row['diff_before'])} "
            f"({_fmt(row['boot_lo_before'])} to {_fmt(row['boot_hi_before'])}, {before_word} zero) "
            f"became {_fmt(row['diff_after'])} ({_fmt(row['boot_lo_after'])} to {_fmt(row['boot_hi_after'])}, {after_word} zero). "
            f"Baseline posts {int(row['n_baseline_before'])} to {int(row['n_baseline_after'])}."
        )
    if its_before.exists() and its_after.exists():
        its_moved = moved_rows(pd.read_csv(its_before), pd.read_csv(its_after))
        for _, row in its_moved.iterrows():
            before_word = "excluded" if row["excluded_before"] else "included"
            after_word = "excludes" if row["excluded_after"] else "includes"
            lines.append(
                f"- Interrupted series, {row['event_id']}, {row['audience']}: {_fmt(row['post_before'])} "
                f"({_fmt(row['boot_lo_before'])} to {_fmt(row['boot_hi_before'])}, {before_word} zero) "
                f"became {_fmt(row['post_after'])} ({_fmt(row['boot_lo_after'])} to {_fmt(row['boot_hi_after'])}, {after_word} zero)."
            )
    return lines


def _lookup_share(shares: pd.DataFrame, audience: str, stage: str) -> str:
    if shares.empty:
        return ""
    hit = shares[(shares["audience"] == audience) & (shares["stage"] == stage)]
    if hit.empty:
        return ""
    return f"{_fmt(hit.iloc[0]['share'])} (n={int(hit.iloc[0]['k'])})"


def _baseline_stage_table(shares: pd.DataFrame) -> list[str]:
    if shares.empty:
        return ["The small-model stage table has not been written yet."]
    stages = list(dict.fromkeys(shares["stage"].tolist()))
    lines = ["| Stage on the small model | Enthusiasts | Shoppers |", "|---|---:|---:|"]
    for stage in stages:
        lines.append(f"| {stage} | {_lookup_share(shares, 'enthusiast', stage)} | {_lookup_share(shares, 'shopper', stage)} |")
    return lines


def _baseline_rdit_table(rdit: pd.DataFrame) -> list[str]:
    if rdit.empty:
        return ["The discontinuity table has not been written yet."]
    rows = rdit[rdit["fit"].isin(["main", "donut"])].copy()
    lines = [
        "| Cutoff | Audience | Outcome | Fit | Bandwidth (days) | Jump | Posts before | Posts after | Thin |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"| {row['cutoff_id']} | {row['audience']} | {row['outcome']} | {row['fit']} | {int(row['bandwidth_days'])} | "
            f"{_fmt(row['tau'])} | {int(row['n_left'])} | {int(row['n_right'])} | {row['thin']} |"
        )
    return lines


def _baseline_author_lines(authors: pd.DataFrame) -> list[str]:
    if authors.empty:
        return ["Author coverage has not been written yet."]
    lines = [
        "| Audience | Posts with a hash | Authors | Authors with 2 or more posts | Share of posts from those authors | Median posts | Max posts |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in authors.iterrows():
        lines.append(
            f"| {row['audience']} | {int(row['n_posts'])} | {int(row['n_authors'])} | {int(row['n_repeat_authors'])} | "
            f"{_fmt(row['share_posts_from_repeat_authors'])} | {_fmt(row['median_posts_per_author'], 1)} | {int(row['max_posts_per_author'])} |"
        )
    return lines


def render_manuscript(out_dir) -> str:
    out_dir = Path(out_dir)
    overall = pd.read_csv(out_dir / "framing_overall_shares.csv") if (out_dir / "framing_overall_shares.csv").exists() else pd.DataFrame()
    boot = pd.read_csv(out_dir / "framing_block_bootstrap.csv") if (out_dir / "framing_block_bootstrap.csv").exists() else pd.DataFrame()
    its = pd.read_csv(out_dir / "framing_its_bootstrap.csv") if (out_dir / "framing_its_bootstrap.csv").exists() else pd.DataFrame()
    manifest = {}
    if (out_dir / "thin_baseline_manifest.json").exists():
        manifest = json.loads((out_dir / "thin_baseline_manifest.json").read_text())
    n_enth = ""
    n_shop = ""
    if not overall.empty:
        enth = overall[overall["audience"] == "enthusiast"]
        shop = overall[overall["audience"] == "shopper"]
        if len(enth):
            n_enth = str(int(enth.iloc[0]["n"]))
        if len(shop):
            n_shop = str(int(shop.iloc[0]["n"]))
    appended = manifest.get("n_appended", "")
    cap = manifest.get("max_posts", 3360)
    minutes = manifest.get("label_minutes", "")
    shares = pd.read_csv(out_dir / "funnel_stage_shares.csv") if (out_dir / "funnel_stage_shares.csv").exists() else pd.DataFrame()
    rdit = pd.read_csv(out_dir / "rdit.csv") if (out_dir / "rdit.csv").exists() else pd.DataFrame()
    authors = pd.read_csv(out_dir / "author_coverage.csv") if (out_dir / "author_coverage.csv").exists() else pd.DataFrame()
    lines = [
        "# Stage and topic in electric-vehicle talk on Reddit, 2015–2026",
        "",
        "**Draft for Transportation Research Part D. Every number in the baseline section is unvalidated output from the small zero-shot model. No human has coded the annotation sheet. The instruction-model results section is a placeholder. Do not cite either as a finding about buyers.**",
        "",
        "## Title options",
        "",
        "1. Where the barrier sits in the purchase: stage and topic in electric-vehicle talk on Reddit, 2015–2026.",
        "2. Enthusiast and shopper submissions around four US electric-vehicle shocks: a six-stage funnel.",
        "3. Document-level stage, topic family, and barrier labels for electric-vehicle discussion on Reddit.",
        "",
        "Working title used below: option 1.",
        "",
        "## Abstract",
        "",
        "Surveys of people who say they might buy an electric vehicle record price, range, and charging as barriers. They miss the month a credit or a price cut arrives, and they miss the wording used when nobody is asking. This paper reads Reddit submissions from 2015 through September 2026. Each submission gets one stage in a six-stage purchase funnel, one or more of eight topic families, a finer barrier aspect when one is present, and a stance. The stage is the writer's place in the journey, not a keyword. Enthusiast posts are a week sample of seven electric-vehicle subreddits. Shopper posts are keyword-matched submissions from r/cars, r/Autos, and r/carbuying. The corpus is labeled by an open instruction model of about 7 to 8 billion parameters, run locally. A smaller zero-shot model is the baseline, not the instrument. Two annotators code 800 posts, including a 50-post calibration subset that is left out of kappa. That sheet is still blank, so this draft has no validated result.",
        "",
        f"On the small-model baseline the enthusiast file has {n_enth} posts and the shopper file has {n_shop}. That model has one ownership class, so it cannot separate onboarding from long-term ownership. The discontinuity and the author-transition counts in section 7 are the same baseline. They are not effects. Section 8 is empty until the local model has been run.",
        "",
        "## 1. Introduction and research questions",
        "",
        "Electric-vehicle policy treats price, charging, and information as what keeps households out. Those claims are usually tested with surveys of people who already say they might buy. A survey misses the language used when the writer was not recruited, and it misses the particular day a credit rule or a price cut arrives.",
        "",
        "Reddit is a public record of that language. Ruan and Lv (2022) recovered topics and a sentiment trend from 3,437,917 Reddit posts, mostly comments, from January 2011 through December 2020. Ruan and Lv (2023) showed that the Reddit pattern over that decade is not the Twitter pattern. Ashby, Weir, and Fussey (2025) organized charging talk on just under 100,000 tweets. Yu et al. (2025) showed what a document-level check looks like, on charging-station reviews rather than on Reddit. None of those papers measures, at the post and with a double-coded check, where the writer is in the purchase and what the post is about, through the dates that now organize US electric-vehicle buying.",
        "",
        "The paper is built on a funnel. A post can be about charging at two different moments: while the writer is still deciding, and in the first week of ownership. Collapsing those into one topic, or into one ownership stage, hides the policy question. The taxonomy in `v2/docs/reference_taxonomy_6stage_8family.pdf` keeps six stages, with onboarding separate from long-term ownership, and eight topic families. The seven barriers already used in this project stay as the finer aspects.",
        "",
        "The communities are part of the question, and a subreddit is not a person. Posts in r/electricvehicles, r/teslamotors, r/BoltEV, r/Rivian, r/Ioniq5, r/KiaEV6, and r/evcharging are the enthusiast file. Posts in r/cars, r/Autos, and r/carbuying that mention an electric vehicle are the shopper file. Owners post in r/cars, and shoppers post in r/teslamotors. The stage label, not the subreddit name, is what places a post in the funnel.",
        "",
        "Four research questions:",
        "",
        "1. How do topic-family and barrier shares differ across the six stages, and do those stage-by-family patterns differ between the enthusiast file and the shopper file?",
        "2. At each of four cutoffs, does the share of posts about the pre-specified barrier jump: the Inflation Reduction Act on 16 August 2022 (incentives), the credit rules for vehicles placed in service after 31 December 2022 (incentives, cutoff 1 January 2023), Tesla's US price cuts on 13 January 2023 (price), and the end of the federal clean-vehicle credit on 30 September 2025 (incentives)?",
        "3. Among authors who post more than once, how often does the next post sit further down the funnel, and is that rate higher for pairs that cross one of the four cutoffs?",
        "4. After two people code the 800-post sheet, do the local instruction model and the small baseline model agree with the agreed human labels well enough to support the first three questions?",
        "",
        "Volume is reported next to the shares. A change in how many posts were made is not a change in what the posts were about. A jump at a cutoff is not, by itself, an effect of the statute or the price cut. Hausman and Rapson (2018) are the reason to say so: a regression discontinuity in time is closer to an event study than to a cross-sectional discontinuity, because other news can land on the same day and because the bandwidth does not shrink in the way a cross-sectional design assumes.",
        "",
        "The hypotheses that match those questions are in `v2/paper/introduction.md`. They are not tested in this draft.",
        "",
        "## 2. Related work",
        "",
        "### 2.1 Surveys of adoption barriers",
        "",
        "Price, range, charging access, and information are the usual survey list. The survey papers already in use for this project belong here. [CITATION NEEDED]",
        "",
        "### 2.2 Social media as a record of electric-vehicle opinion",
        "",
        "Ruan and Lv (2022) fit a 20-topic LDA model and compared LIWC, VADER, and AFINN on Reddit through December 2020. The article is in *Communications in Transportation Research*, not Transportation Research Part D. Ruan and Lv (2023) repeat that corpus and add 7,383,327 tweets. Ashby, Weir, and Fussey (2025) use BERTopic and a language model on charging tweets, with manual review of the theme list rather than a double-coded test of document labels. The unit in the present paper is the submission. Ruan and Lv's mass is mostly comments, so a comparison of levels is not like-for-like.",
        "",
        "### 2.3 Language models as annotators",
        "",
        "Gilardi, Alizadeh, and Kubli (2023) report that zero-shot ChatGPT beat crowd workers by about 25 percentage points on average across four samples (n = 6,183), and they treat validation as part of the method. Yu et al. (2025) double-coded sentiment on 3,188 charging-station reviews (Cohen's κ = 0.81) and problem categories (human κ = 0.71); language-model accuracy on the categories was 0.77 to 0.82. Ha, Marchetto, Dharur, and Asensio (2021) train a transformer on expert-annotated charging reviews. That paper is the template for a supervised classifier after a gold set exists. An accuracy figure for Ha et al. is not repeated here, because it was not re-checked against the publisher page in this project. VADER is named by Ruan and Lv (2022); the primary VADER paper was not opened here. [CITATION NEEDED]",
        "",
        "Cohen's kappa, Krippendorff's alpha, and the Rogan-Gladen prevalence correction are the agreement statistics the scoring script computes. The bibliographic entries for those three methods were not opened in this project. [CITATION NEEDED]",
        "",
        "### 2.4 Discontinuity in time",
        "",
        "Hausman and Rapson (2018), *Annual Review of Resource Economics* 10:533–552, https://doi.org/10.1146/annurev-resource-121517-033306, set out how a regression discontinuity in time differs from a cross-sectional discontinuity: estimation often uses observations far from the cutoff, serial correlation can bias the jump, and a test for sorting around the cutoff is often beside the point. This paper uses that framing. It does not add a second methods citation that has not been opened. The cross-sectional bandwidth literature is [CITATION NEEDED].",
        "",
        "## 3. Data",
        "",
        "Counts of enthusiast submissions come from the Arctic Shift API (`https://arctic-shift.photon-reddit.com`). Pushshift no longer updates, and the official Reddit API needs credentials this project does not have. Complete months from January 2015 through September 2026 contain 603,435 submissions in the seven enthusiast subreddits. October 2026 was partial on 16 October 2026 (2,440 submissions).",
        "",
        "The enthusiast text file is not that census. It is one deterministic 7-day window per subreddit-quarter, limit 25, plus a fixed week around each coded event, limit 40. The window rule is in `v2/download_reddit.py`. Round 6 added three further 7-day windows, limit 40, inside four baseline months that the week sample had left thin: August 2021, June 2022, December 2022, and September 2024. "
        f"The cap on that fill was {cap} posts and 90 minutes of labeling"
        + (f"; it appended {appended} posts in {minutes} minutes." if appended != "" else "."),
        "",
        "Shopper text is streamed from Academic Torrents zstd dumps (Watchful1, 2026, subreddit submissions 2005-06 to 2025-12, infohash `3e3f64dee22dc304cdd2546254ca1f8e8ae542b4`). The keyword filter is a word-boundary match for tesla, rivian, ioniq, \"electric vehicle\", and \"electric car\". Bare `ev` is excluded. The extract through December 2025 kept 13,976 posts. January–September 2026 kept 421 keyword posts, of which 413 survived cleaning. The API aggregate for the same three subreddits and the same keywords, January 2015 through September 2026, is 16,993. The two matchers are different, and the dump filter was not retuned to force the totals together. Comments were not downloaded. Submissions only is a locked decision.",
        "",
        f"After cleaning, the labeled analysis file has {n_enth} enthusiast posts and {n_shop} shopper posts. A probable bot, or a post under five tokens, is dropped before labeling. Usernames are not in the committed label file. The annotation sheet redacts `u/` mentions. Full post text stays in gitignored `v2/data/raw/`.",
        "",
        "The license field on the Academic Torrents item is empty. A takedown thread exists. The paper can report counts and the measurement method. It should not republish the dumps.",
        "",
        "## 4. Measurement",
        "",
        "### 4.1 The local instruction model, and the small model as baseline",
        "",
        "The instrument for the corpus is an open instruction model in the 7 to 8 billion parameter class, run on a local RTX 4070 with 16 GB of memory. The recommended first checkpoint is Qwen2.5-7B-Instruct in 4-bit. Llama-3.1-8B-Instruct is the alternative that also fits. No paid API is used. The prompt is the codebook: one stage, one or more families, the finer aspects, a stance, and a short quote from the post. The output is JSON. Each row stores the model name, the model version, and a hash of the prompt, so a later paper can compare the small model with the larger one without mixing the files. The runner is `v2/llm_label.py`. The steps for the desktop are `v2/RUN_ON_MY_GPU.md`. A 100-post smoke file is `v2/sample/smoke_100.csv`.",
        "",
        "The small model stays as the baseline and is the only model that has labeled the 22,827 posts so far. It is `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`, revision `262ae02f29173eec1c250f90804dc7edc677dcff`, on CPU. Barriers and stage use the template \"This text is about {}.\". Stance uses \"This text expresses {}\". A barrier is on when its score is at least 0.5. Dealer friction is also on when a small lexicon matches a markup, a documentation fee, a market adjustment, or a dealer inventory wait, and it is turned off when the model fires on ordinary service or on a charger that merely sits at a dealer and the lexicon did not match. That lexicon check is on invented sentences (`v2/fixtures/dealer_sentences.csv`). The small model has one ownership class. It does not emit the eight families. It is not replaced, and it is not the result.",
        "",
        "### 4.2 Taxonomy",
        "",
        "The codebook is `v2/annotation/CODEBOOK.md`. Stage is one code: awareness, consideration, evaluation, purchase, onboarding, long-term ownership, or unclear. Onboarding is the first period of ownership (setup, the app, the first week). Long-term ownership is later use (degradation, warranty, service, resale, regret). Charging can be either, depending on whether the writer already has the car. Families are multi-label: purchase and vehicle selection; charging and energy access; battery, range, and real-world performance; cost, incentives, and financial feasibility; dealer, purchase process, and market access; reliability, service, and ownership experience; policy, regulation, and market environment; and ICE, hybrid, and EV transition narratives. Aspects are the seven barriers: charging access, driving range, price or total cost, purchase incentives or tax credits, battery reliability, dealer or buying friction, and environmental impact, plus an explicit none. Stance is support, oppose, mixed, or no stance. A complaint about one broken charger is not opposition by itself.",
        "",
        "### 4.3 Annotation and validation",
        "",
        "The sheet is `v2/annotation/annotation_sheet.csv`: 800 posts (382 enthusiast, 418 shopper), excerpts of at most 500 characters, code cells empty. The draw covers all six stages and all eight families in each audience, with rare cells filled first. A short lexicon was used only to place some of the small model's ownership posts into the onboarding pool and some into the long-term pool, so both stages appear on the sheet. That lexicon is not a code, and it is not printed on the sheet. Fifty rows are flagged `calibration=yes`. The two annotators code those first, compare them, and freeze the codebook. The 50 are excluded from kappa. The practice rows in `annotation_preview.csv` are also excluded. Model predictions are in `annotation_model_key.csv`, not on the sheet. Daisy is one annotator. The second is a labmate. This project did not contact anyone.",
        "",
        "`v2/score_agreement.py` reads the two finished sheets. It writes Cohen's kappa and Krippendorff's alpha for each aspect, each family, stage, and stance, with a bootstrap interval for kappa. It scores a model only against items both coders marked and agreed on, and it leaves disagreements out of that gold set. It writes precision, recall, and F1, and a Rogan-Gladen correction of the prevalence. Blanks are missing, not a third category. The calibration rows are written to their own table and left out of the main one. The script was run on synthetic labels only. It does not invent a coder.",
        "",
        "The rule already fixed is: if the lower bound of the confidence interval on the aspect kappas is below 0.60, add 200 adjudicated posts before a substantive claim, and if it is still too low, hold the Transportation Research Part D submission. The same comparison is then run for the small model and for the local instruction model. `v2/run_after_llm.py` does that second step when both coder files are passed in. Until those files exist, model-versus-human accuracy is not a number in this paper.",
        "",
        "## 5. Discontinuity in time",
        "",
        "### 5.1 Estimand",
        "",
        "For a cutoff date c and a pre-specified aspect, let Y_i be 1 when post i is labeled with that aspect and 0 otherwise. Let X_i be the post's calendar day minus c, in days. The estimand is the jump in the conditional mean at the cutoff:",
        "",
        "tau = limit as x approaches 0 from the right of E[Y | X = x] minus the limit as x approaches 0 from the left of E[Y | X = x].",
        "",
        "Posts are aggregated to the day. Day t has n_t posts and share Y-bar_t. A local linear regression is fit separately on each side of the cutoff, with a triangular kernel K(x) = max(0, 1 - |x| / h), and with weight n_t K(x). tau-hat is the right-hand intercept minus the left-hand intercept. The outcome is incentives at the two credit cutoffs and price at the Tesla cutoff. The other aspects are not the pre-specified outcomes.",
        "",
        "The 1 January 2023 cutoff is twelve days before 13 January 2023. The two fits are not separate experiments. Hausman and Rapson (2018) are also the reason not to read tau as the effect of the statute: another announcement can fall on the same day, and the enthusiast file is a week sample, so many days are empty.",
        "",
        "### 5.2 Bandwidth, donut, and placebos",
        "",
        "The bandwidth h is chosen from the grid 30, 60, 90, and 180 days by even/odd day cross-validation. Even days are used to fit the two boundary levels. Odd days are predicted from those levels, without a slope. The bandwidth with the lowest weighted squared error is the main one. The other three bandwidths are reported as a robustness grid and are not re-labeled as the main result.",
        "",
        "The donut refits the chosen bandwidth after dropping posts with |X| at most 7 days. A thin fit has fewer than 30 posts or fewer than 5 distinct days on either side. A thin fit is reported and not interpreted.",
        "",
        "Placebo cutoffs use the same outcome and the same chosen bandwidth, moved by 180 days earlier and 180 days later. A placebo that is as large as the main jump is a reason to doubt the main jump. It is not a second finding.",
        "",
        "### 5.3 Inference",
        "",
        "The interval is a percentile interval from a week-cluster bootstrap. Weeks inside the bandwidth are resampled with replacement, the daily shares are rebuilt, and the local linear jump is refit at the bandwidth chosen on the original sample. The bandwidth is not re-chosen inside each draw, so the interval is for that bandwidth. The main fit and the donut are resampled. The placebo and the bandwidth grid are point estimates. `v2/run_after_llm.py` writes the interval when it is pointed at a label file. The small-model table in section 7 is the point estimate only. An interval was not attached to that baseline table.",
        "",
        "## 6. Stage transitions",
        "",
        "Authors are hashed. Deleted accounts are dropped. The hash is not a name. Posts by the same hash are ordered by time. A pair is two successive posts. On the six-stage ladder the rungs are awareness, consideration, evaluation, purchase, onboarding, and long-term ownership, in that order. Unclear is not on the ladder. A pair is a forward move when both posts sit on the ladder and the later rung is higher. The rate is the share of pairs that move forward.",
        "",
        "For a cutoff c, the shock rate uses pairs whose first post falls in the 180 days before c and whose second post falls in the 180 days after c. The same window is applied at c minus 180 days and at c plus 180 days. The comparison is descriptive. It is not a hazard model with covariates, and it is not an estimate of how fast a person moves from shopping to buying. The small-model baseline cannot see onboarding versus long-term ownership, so its ladder stops at one ownership rung. The instruction-model file is the one that can use the six-rung ladder. Repeat-author coverage is the feasibility check. A week sample contains the same person twice less often than a census would.",
        "",
        "## 7. Preliminary baseline from the small model",
        "",
        "**These numbers are the small zero-shot model. They are not the instruction-model results. They will move when that model is run, and they will move again when human labels exist. They are not a result about buyers.**",
        "",
        "### 7.1 Stage shares",
        "",
        "Ownership is one class on this model. Onboarding is not identified. The denominator is the labeled file for that audience.",
        "",
    ]
    lines.extend(_baseline_stage_table(shares))
    lines.extend(
        [
            "",
            "### 7.2 Finer aspects",
            "",
            "The aspect shares below are the small model's barrier indicators. The eight families are not in this file.",
            "",
        ]
    )
    if not overall.empty:
        barrier_names = {
            "charging access",
            "driving range",
            "vehicle price or total cost",
            "purchase incentives or tax credits",
            "battery reliability",
            "dealer or buying friction",
            "environmental impact",
        }
        lines.extend(_share_table(overall[overall["measure"].isin(barrier_names)]))
    lines.extend(
        [
            "",
            "### 7.3 Discontinuity, point estimates only",
            "",
            "Local linear jump in the daily share. Bandwidth chosen on the grid in section 5. No week-cluster interval is attached to this baseline table. The Tesla enthusiast donut is unstable and is not interpreted.",
            "",
        ]
    )
    lines.extend(_baseline_rdit_table(rdit))
    lines.extend(
        [
            "",
            "### 7.4 Whether authors post twice",
            "",
        ]
    )
    lines.extend(_baseline_author_lines(authors))
    lines.extend(
        [
            "",
            "Pairs that cross a cutoff inside 180 days number about 100 on this file, and about 39 at the credit end. Forward rates on the unsplit ladder are about 0.13 to 0.18, and the placebos 180 days away fall in the same range. The median author has one post. A few hashes contribute many posts, so the repeat-post share is not a panel of distinct buyers. The counts are in `v2/outputs/stage_transition_shocks.csv`.",
            "",
            "An earlier descriptive contrast, event-window share minus a baseline, with a design-aware bootstrap, is in `v2/outputs/FINDINGS.md`. It is not the estimand in section 5.",
            "",
            "## 8. Instruction-model results",
            "",
            "**PLACEHOLDER. No instruction-model labels have been produced. Do not copy the small-model numbers into this section. Do not fill it from the mock backend used to test the pipeline.**",
            "",
            "When `v2/outputs/llm_labels.csv` exists, this command rebuilds the section:",
            "",
            "`python v2/run_after_llm.py --llm v2/outputs/llm_labels.csv --out v2/outputs/after_llm`",
            "",
            "It writes funnel shares by audience, a stage-by-family heatmap, the discontinuity table with the bandwidth grid, the donut, the placebos, and the week-cluster interval, the stage-transition rates on the six-rung ladder, and agreement between the small model and the instruction model. Agreement with humans is written only when both annotator sheets are passed as `--coder-a` and `--coder-b`. If those files are absent, the command records that they were not provided. It does not invent them.",
            "",
            "The cells that belong here, and that are empty on purpose:",
            "",
            "| Result | Status |",
            "|---|---|",
            "| Stage shares by audience, six stages | Not run |",
            "| Stage-by-family heatmap, by audience | Not run |",
            "| Discontinuity jumps, with bandwidth grid, donut, placebos, and week-cluster intervals | Not run |",
            "| Forward-move rates on the six-rung ladder, with placebos | Not run |",
            "| Small-model versus instruction-model agreement | Not run |",
            "| Human kappa and model-versus-human accuracy | Not run. The sheet is blank. |",
            "",
            "## 9. Limitations",
            "",
            "English-language Reddit is not the US household population. The enthusiast file is a cluster sample of submissions, not the 603,435-post census. The shopper file is keyword-filtered. Comments are absent. Arctic Shift does not guarantee that the archive is complete. The Academic Torrents license field is empty. Other news happens on the same days as the four cutoffs, so a jump is not an effect. The 1 January 2023 and 13 January 2023 cutoffs are twelve days apart. The small model cannot split ownership. The instruction model has not been run. Labels have not been checked by a person.",
            "",
            "## 10. What remains before submission",
            "",
            "1. The labmate codes `v2/annotation/annotation_sheet.csv` against `v2/annotation/CODEBOOK.md`, calibration rows first. The name is not written here.",
            "2. Run the 100-post smoke test in `v2/RUN_ON_MY_GPU.md`, then the full local model, then `v2/run_after_llm.py`.",
            "3. Run `v2/score_agreement.py` on the two coder files. If the lower bound on aspect kappa is below 0.60, code 200 more posts before any substantive sentence.",
            "4. Replace section 8 with the command's tables. Keep section 7 labeled as the small-model baseline.",
            "5. Comments stay out. An upvote filter (score at least 2) has not been run. If it is adopted later, show it as a sensitivity.",
            "6. Resolve every `[CITATION NEEDED]` mark. Do not add a DOI that has not been opened.",
            "7. Write the data-availability and ethics paragraphs the journal asks for: public posts, no attempt to identify users, dumps not republished, license field empty. Check the journal's current author guide for length and reference style. This draft does not guess those limits.",
            "",
            "## References",
            "",
            "Ashby, C., Weir, D., and Fussey, P. (2025). Understanding public views on electric vehicle charging: A thematic analysis. *Transportation Research Interdisciplinary Perspectives*, 29, 101325. https://doi.org/10.1016/j.trip.2025.101325",
            "",
            "Gilardi, F., Alizadeh, M., and Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *PNAS*, 120(30), e2305016120. https://doi.org/10.1073/pnas.2305016120",
            "",
            "Ha, S., Marchetto, D. J., Dharur, S., and Asensio, O. I. (2021). Topic classification of electric vehicle consumer experiences with transformer-based deep learning. *Patterns*, 2, February 2021. https://doi.org/10.1016/j.patter.2020.100195",
            "",
            "Hausman, C., and Rapson, D. S. (2018). Regression discontinuity in time: Considerations for empirical applications. *Annual Review of Resource Economics*, 10, 533–552. https://doi.org/10.1146/annurev-resource-121517-033306",
            "",
            "Ruan, T., and Lv, Q. (2022). Public perception of electric vehicles on reddit over the past decade. *Communications in Transportation Research*, 2, 100070. https://doi.org/10.1016/j.commtr.2022.100070",
            "",
            "Ruan, T., and Lv, Q. (2023). Public perception of electric vehicles on Reddit and Twitter: A cross-platform analysis. *Transportation Research Interdisciplinary Perspectives*, 21, 100872. https://doi.org/10.1016/j.trip.2023.100872",
            "",
            "Yu, Q., Que, T., Cushing, L. J., Pierce, G., Shen, K., Kejriwal, M., Yao, Y., and Zhu, Y. (2025). Equity and reliability of public electric vehicle charging stations in the United States. *Nature Communications*. https://doi.org/10.1038/s41467-025-60091-y",
            "",
            "Watchful1. (2026). Subreddit comments/submissions 2005-06 to 2025-12 [Data set]. Academic Torrents. Infohash `3e3f64dee22dc304cdd2546254ca1f8e8ae542b4`. The license field on the torrent is empty.",
            "",
            "Event dates, as used in `v2/research_design.md`: Pub. L. 117-169; 26 U.S.C. § 30D; Reuters, 13 January 2023, https://www.reuters.com/business/autos-transportation/tesla-cuts-prices-electric-vehicles-us-market-2023-01-13/; Reuters, 25 May 2023, https://www.reuters.com/business/autos-transportation/tesla-ford-ceos-talk-evs-twitter-forum-2023-05-25/; General Motors news release, 8 June 2023, https://investor.gm.com/news-releases/news-release-details/general-motors-doubles-down-commitment-unified-charging-standard/; Pub. L. 119-21; IRS, https://www.irs.gov/clean-vehicle-tax-credits.",
            "",
            "Hutto and Gilbert (2014), Cohen, Krippendorff, Rogan and Gladen, and the cross-sectional regression-discontinuity bandwidth papers: [CITATION NEEDED].",
            "",
            "Survey literature on adoption barriers: [CITATION NEEDED].",
            "",
        ]
    )
    return "\n".join(lines)

_NUMBERED = re.compile(r"^\d+\. ")


def markdown_to_tex(markdown: str) -> str:
    """A small subset so the draft can be opened in a TeX editor without pandoc."""
    out = [
        "\\documentclass[11pt]{article}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage{booktabs}",
        "\\usepackage{hyperref}",
        "\\usepackage{longtable}",
        "\\title{Stage and topic in electric-vehicle talk on Reddit, 2015--2026}",
        "\\author{Draft. Results are preliminary and unvalidated.}",
        "\\date{}",
        "\\begin{document}",
        "\\maketitle",
        "",
    ]
    table_rows: list[list[str]] = []
    bullets: list[str] = []
    numbers: list[str] = []

    def flush_table() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        # Drop the markdown separator row.
        body = [row for row in table_rows if not all(set(cell) <= set("-: ") for cell in row)]
        if not body:
            table_rows = []
            return
        ncol = max(len(row) for row in body)
        out.append("\\begin{longtable}{" + "l" * ncol + "}")
        out.append("\\toprule")
        for i, row in enumerate(body):
            cells = row + [""] * (ncol - len(row))
            out.append(" & ".join(_tex_escape(cell) for cell in cells) + " \\\\")
            if i == 0:
                out.append("\\midrule")
        out.append("\\bottomrule")
        out.append("\\end{longtable}")
        out.append("")
        table_rows = []

    def flush_bullets() -> None:
        nonlocal bullets
        if not bullets:
            return
        out.append("\\begin{itemize}")
        for item in bullets:
            out.append("\\item " + _tex_inline(item))
        out.append("\\end{itemize}")
        out.append("")
        bullets = []

    def flush_numbers() -> None:
        nonlocal numbers
        if not numbers:
            return
        out.append("\\begin{enumerate}")
        for item in numbers:
            out.append("\\item " + _tex_inline(item))
        out.append("\\end{enumerate}")
        out.append("")
        numbers = []

    def flush_lists() -> None:
        flush_bullets()
        flush_numbers()

    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            flush_lists()
            table_rows.append([cell.strip() for cell in line.strip("|").split("|")])
            continue
        flush_table()
        if line.startswith("# "):
            flush_lists()
            continue
        if line.startswith("## "):
            flush_lists()
            out.append("\\section{" + _tex_escape(line[3:]) + "}")
        elif line.startswith("### "):
            flush_lists()
            out.append("\\subsection{" + _tex_escape(line[4:]) + "}")
        elif line.startswith("- "):
            flush_numbers()
            bullets.append(line[2:])
        elif _NUMBERED.match(line):
            flush_bullets()
            numbers.append(line.split(".", 1)[1].strip())
        elif line == "":
            flush_lists()
            out.append("")
        else:
            flush_lists()
            out.append(_tex_inline(line))
            out.append("")
    flush_table()
    flush_lists()
    out.append("\\end{document}")
    out.append("")
    return "\n".join(out)


def _tex_escape(text: str) -> str:
    for char, repl in (
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
    ):
        text = text.replace(char, repl)
    return text


def _tex_inline(text: str) -> str:
    # Bold markers, then escape the remainder in pieces.
    parts = text.split("**")
    rendered = []
    for i, part in enumerate(parts):
        code_bits = part.split("`")
        piece = []
        for j, bit in enumerate(code_bits):
            escaped = _tex_escape(bit)
            piece.append("\\texttt{" + escaped + "}" if j % 2 == 1 else escaped)
        joined = "".join(piece)
        rendered.append("\\textbf{" + joined + "}" if i % 2 == 1 else joined)
    return "".join(rendered)


def write_manuscript(out_dir, paper_dir) -> None:
    paper_dir = Path(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_manuscript(out_dir)
    (paper_dir / "manuscript.md").write_text(markdown)
    (paper_dir / "manuscript.tex").write_text(markdown_to_tex(markdown))
