# What remains before submission

Updated 30 October 2026, after Ryan's 50 calibration rows were scored. Nothing here is a new label. The codebook, the prompts, and the manuscript results were not changed.

The draft is `manuscript.md`. Submissions are the main specification. Comments v2 are a sensitivity in section 8.7. Every model number is unvalidated.

## Still open

1. **Human kappa.** Ryan uploaded `v2/annotation/annotation_sheet_ryan.xlsx` (commit `f98c050`) and coded only the 50 calibration rows. Daisy scored those rows against the models in `v2/outputs/human_agreement/`. On that set, stage kappa versus 14B is 0.16 [0.02, 0.30] and pooled family kappa is 0.44. That agreement is low. The codebook excludes the calibration rows from the paper's kappa, so section 8 still has an empty human-kappa cell. There is no Daisy-versus-Ryan kappa yet. Next: Daisy codes the same 50 rows, they settle the codebook wording (news and reviews count as unclear, and owner experience without a just-bought cue counts as long-term ownership), then Ryan codes the remaining 750. The README still names `v2/annotation/coder_ryan.csv` as the return file; this first pass arrived as the xlsx. Instructions remain `v2/annotation/README_FOR_RYAN.md`. This project has not contacted him.

2. **`performance` at the codebook freeze.** `aspect:performance` is the one invented code that is not a level mixup (252 rows in the submission run). It stays out of the codebook until Ryan and Daisy freeze the sheet. Do not add it in a silent edit, and do not relabel for it before that freeze. The recommendation is in `v2/COMMENTS_V2_PLAN.md`.

3. **Citations.** The marks in the draft body are resolved. `references.bib` holds the DOIs checked against Crossref on 30 October 2026: the survey and review papers (Egbue and Long 2012; Rezvani, Jansson, and Bodin 2015; Coffman, Bernstein, and Wee 2017), Hutto and Gilbert 2014, Cohen 1960, Krippendorff 2004, Rogan and Gladen 1978, and the cross-sectional bandwidth papers (Imbens and Kalyanaraman 2012; Calonico, Cattaneo, and Titiunik 2014), plus the sources the draft already named. No `[CITATION NEEDED]` mark remains in `manuscript.md`. One claim was softened instead of cited: a separate paper that isolates an information barrier, as distinct from the knowledge difference in Egbue and Long, was not opened. Do not add a DOI that has not been opened. The alternative title is in `ALT_FRAMING.md` and is not the working title.

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

Decision needed: whether the title and the abstract keep the four shocks in the lead, or move the shocks to a sensitivity and lead with the funnel description and the measurement limits. A draft of that alternative is `ALT_FRAMING.md`. It is not adopted. The working title and the abstract in `manuscript.md` are unchanged.
