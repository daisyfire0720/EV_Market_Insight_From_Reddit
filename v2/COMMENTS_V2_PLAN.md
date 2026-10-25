# Comments v2: a design, not a run

This note does not pull comments, does not call a model, and does not change the codebook. The 14B labels already in `outputs/llm_labels.csv` stay as they are. Submissions remain the main specification.

Batch 1 (26,088 comments) was labeled without the parent post. About 65% of enthusiast comments and about 85% of shopper comments are `unclear`. The batch was also picked by closeness to a cutoff. Inside 180 days of the Tesla cuts, enthusiast comments with a codebook stage are 12,022 before the day, 1,495 on the day, and 1,470 after (`outputs/after_llm/comment_window_balance.csv`). An uncommitted exploratory count in `outputs/after_llm/FINDINGS_14B.md` was 13,374 left and 2,290 right. Either count is lopsided. `comments_sensitivity.csv` sets `rdit_usable` to false. Do not fit a discontinuity on batch 1.

The rate below is the one given for this note: **0.91 seconds per item** on the same 14B setup (Qwen2.5-14B-Instruct, Q4_K_M, one RTX 4070). Hours are `n × 0.91 / 3600`. No new timing run was done.

## 1. Put the parent title in the prompt

The current prompt prefixes a comment with `[comment]` and then the comment body. The model never sees the post the comment is answering, so a short reply has no journey to label.

Proposed prefix, still one comment per row:

```
[comment]
Parent title: {title}
Parent excerpt: {first 300 characters of the parent body, or empty}
Comment:
{comment body}
```

Rules:

- Use the parent submission's title and body. Do not add a username.
- If the parent is not in the local cache, leave the excerpt empty and keep the title if the metadata has it. Do not invent either.
- Store a new prompt hash. Do not overwrite `outputs/llm_labels.csv`. Write a new cache, for example `data/raw/llm_labels_comments_v2.jsonl`.
- Keep the same codebook, temperature 0, and `qwen2.5:14b-instruct`.

Relabeling the existing 26,088 batch-1 comments would make the stage labels easier to read. It would not balance the windows, so it is not a discontinuity sample.

`26,088 × 0.91 / 3,600 = 6.6 hours.`

## 2. Balanced windows, then label once

Batch 2 as currently queued is 224,706 comments. Labeling that file as it stands would be:

`224,706 × 0.91 / 3,600 = 56.8 hours.`

That spend mostly repeats the closest-to-cutoff pile-up. Do not label batch 2 as queued.

Draw a new sample from comments already inside the ±180-day windows (the metadata in `outputs/comment_meta.csv`, not a new scrape):

- Four cutoffs: 16 August 2022, 1 January 2023, 13 January 2023, 30 September 2025.
- Two audiences, enthusiast and shopper, using the same subreddit lists as the submission file. If the parent submission is in `framing_labels.csv`, use that post's audience instead.
- Two sides: calendar day before the cutoff, and calendar day after. Leave the cutoff day out of both sides so it cannot pad the shorter side.
- Assign each comment to one cutoff only, the nearest one. The 1 January and 13 January cutoffs are twelve days apart, and a comment must not enter both samples.
- Inside each cutoff × audience × side, take a simple random sample. The cap is the same on the left and the right. If one side has fewer comments than the cap, lower the cap to that side's count. Do not fill the long side past the short side.
- Do not rank by distance to the cutoff. The closest-comment rule is what made batch 1 unusable.

Proposed cap: 500 comments per side, per cutoff, per audience, before the short-side trim.

`4 cutoffs × 2 sides × 2 audiences × 500 = 8,000 comments.`

`8,000 × 0.91 / 3,600 = 2.0 hours.`

A cap of 1,000 is 16,000 comments and 4.0 hours, if every side actually has 1,000 comments inside the window. Use 500 unless a side is inspected and is known to be that deep after the nearest-cutoff assignment. This note does not count those cells. The draw has to be counted before anyone books the GPU time, because the short-side rule can only shrink the 8,000.

Label that draw once, with the parent title in the prompt. That is the file that could later be used for a comment discontinuity. Until it exists, comment RDiT stays off.

Relabeling batch 1 (6.6 hours) and labeling the balanced draw (2.0 hours) are separate decisions. The balanced draw is the one that answers the discontinuity. The batch-1 relabel is only a descriptive stage table.

## 3. Whether to add `performance` to the codebook

`aspect:performance` is the one code the model invented that is not a level mixup: 252 rows (`outputs/after_llm/qc_unknown_codes.csv`). The other frequent unknown codes are a family written as an aspect, or the reverse. Those were remapped in `outputs/after_llm_remapped/`. Funnel shares moved by at most 0.2 points and the main jumps by at most 0.3. `performance` was left dropped.

Do not add it in this round.

- Ryan is still coding the sheet against the current codebook. Changing the codebook now splits his codes from the 14B file.
- If it is added later, add it as an aspect, next to range and battery reliability, not as a ninth family. The family list already has "Battery, Range, and Real-World Performance."
- Use the same codebook for submissions and comments. A comments-only code cannot be compared with the submission tables.
- A codebook change needs a new prompt hash and a relabel. The current labeled file is 48,913 rows: `48,913 × 0.91 / 3,600 = 12.4 hours`, plus whatever balanced comment sample is drawn on top.

Recommendation: leave `performance` in `unknown_codes`, keep dropping it, and decide with Ryan when the sheet is frozen. Do not relabel for it before that.

## What not to do

- Do not fit RDiT on batch 1, with or without the parent title.
- Do not pass `--comments-rebalanced` until a file was actually drawn with equal caps on each side.
- Do not label all 224,706 batch-2 comments.
- Do not spend the 12.4-hour full relabel on a silent codebook edit.
