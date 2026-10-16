# v2: measurement pilot

The packages under `src/` are unchanged. This folder is a separate pipeline that:

- downloads submissions from the [Arctic Shift API](https://arctic-shift.photon-reddit.com)
- fixes the five problems called out in the October 2026 review (word-boundary keywords, `topic_id`/`Topic` column names, the funnel input path, the bot-username rule, per-document topic probabilities)
- runs VADER, LDA, and a small zero-shot NLI model on the sample

There is no paid API key and no GPU in the environment this was built on. NLI uses `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` on CPU.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r v2/requirements.txt
```

`torch` is the CPU wheel. The requirements file points pip at `https://download.pytorch.org/whl/cpu`.

## Download

From the repository root:

```bash
python v2/download_reddit.py
```

Writes:

- `v2/data_card/DATA_CARD.md`, `monthly_post_counts.csv`, `yearly_post_counts.csv`, `download_log.json`
- `v2/sample/posts_sample.csv` (small, usernames hashed)
- `v2/data/raw/posts_downloaded.csv` (full text sample, gitignored)

The count step issues one calendar month per request. A year-long aggregate times out on the active subreddits.

## Analyze

```bash
python v2/run_analysis.py
```

Tables and figures land in `v2/outputs/`. An annotation sheet lands in `v2/annotation/`.

## Tests

```bash
cd v2 && pytest
```

## Read first

- `v2/research_design.md` — framing, codebook, validation plan
- `v2/paper/outline.md` — Transportation Research Part D skeleton
- `v2/data_card/DATA_CARD.md` — what was actually downloaded
