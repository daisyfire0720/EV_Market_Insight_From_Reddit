# What remains before submission

Updated 29 October 2026, after the comments v2 sensitivity was folded into the draft. Nothing here is a new label. The codebook was not changed.

The draft is `manuscript.md`. Submissions are the main specification. Comments v2 are a sensitivity in section 8.7. Every model number is unvalidated.

## Still open

1. **Human kappa.** Ryan has not returned `v2/annotation/coder_ryan.csv`. The sheet is `v2/annotation/annotation_sheet.csv`. Instructions are `v2/annotation/README_FOR_RYAN.md`. The kappa table in section 8 leaves that cell as a placeholder. This project has not contacted him. Until both coder files exist, `v2/score_agreement.py` has nothing real to score, and the 14B shares are what a model said.

2. **`performance` at the codebook freeze.** `aspect:performance` is the one invented code that is not a level mixup (252 rows in the submission run). It stays out of the codebook until Ryan and Daisy freeze the sheet. Do not add it in a silent edit, and do not relabel for it before that freeze. The recommendation is in `v2/COMMENTS_V2_PLAN.md`.

3. **Every `[CITATION NEEDED]`.** The draft still marks the survey barrier papers, Hutto and Gilbert (VADER), Cohen, Krippendorff, Rogan and Gladen, and the cross-sectional bandwidth papers. Do not add a DOI that has not been opened.

4. **Journal paragraphs.** Data availability and ethics are not written: public posts, no attempt to identify users, dumps not republished, Academic Torrents license field empty. The journal's length and reference style have not been checked.

## Framing, for Daisy to decide

The two submission jumps that the small model also pointed to are not results a reader should leave with.

- Enthusiast incentive talk at 1 January 2023 is +3.5 points, interval [−5.3, +7.2]. The small model was +5.0. The interval includes zero. Score at least 2 cuts it to +0.5.
- Shopper price talk at the 13 January 2023 Tesla cuts is +10.5 points, interval [−2.2, +24.2]. The small model was +13.1. The interval includes zero. Score at least 2 cuts it to +0.8.
- Comments v2 do not reproduce either jump. On the comments drawn for that cutoff, Tesla shopper price talk is −2.9 [−6.8, +2.7], and the pooled comment window is +0.5 [−5.8, +7.0], against +10.5 in submissions. Enthusiast incentive talk on its own draw is −5.6 [−8.7, +12.7], against +3.5 in submissions.
- Three other submission jumps already fail a fake-date check and stay demoted.
- The inner sides of the two January cutoffs are about six days in the comment draw, because each comment is assigned to only one cutoff. The 7-day donut is empty there. One short-side comment interval excludes zero (shopper incentives at 1 January). That interval is not a result.

A paper that leads with those jumps is leading with estimates that are not distinguishable from zero, that move when low-score posts are dropped, and that the comment sample does not repeat. The part that does not depend on a cutoff is the descriptive split: enthusiast submissions sit at onboarding, shopper submissions sit at awareness, with the stage-by-family grid beside them. That split is still an unvalidated model label.

The parent-title prompt is worth a methods paragraph. Comment stage without the parent post was 65 to 85 percent unclear. With the parent title, enthusiast comments are about 33 percent unclear, close to the submission rate, and shopper comments are still about 67 percent unclear. Batch 1 and comments v2 are different samples, so that drop is not a clean estimate of the prompt. It is a reason to keep the parent in any later comment labeling. It is not a reason to rerun the submission discontinuity.

Decision needed: whether the title and the abstract keep the four shocks in the lead, or move the shocks to a sensitivity and lead with the funnel description and the measurement limits. This draft now states the comment non-replication and the empty donuts. It does not retitle the paper.
