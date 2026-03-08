# EV Reddit BERTopic Pipeline

This project provides Python runners for BERTopic on EV-related Reddit datasets.

## Scripts

- `code/run_ev_reddit.py`
: Runs BERTopic for one subreddit dataset (default: `r/electricvehicles` from `data/data_evforum`).

- `code/run_ev_other.py`
: Runs BERTopic for EV-related content across multiple non-EV subreddits in `data/data_other`.

- `code/run_ev_all.py`
: Runs BERTopic across all EV files in `data/data_all` and supports per-subreddit source-tag mapping.

Core processing lives in `code/ev_bertopic/`.

## Run

Run from the `code/` folder:

```bash
python run_ev_reddit.py
python run_ev_other.py
python run_ev_all.py
```

Examples:

```bash
# Single subreddit dataset
python run_ev_reddit.py --submissions ../data/data_evforum/electricvehicles_submissions.csv --comments ../data/data_evforum/electricvehicles_comments.csv

# Subset of "other" subreddits
python run_ev_other.py --subreddits carbuying autos

# All datasets with source-tag mapping overrides
python run_ev_all.py --source-tag electricvehicles=evforum carbuying=carbuying
```

## Key Outputs

Each runner writes 3 CSV files to its `--output-dir`:

- `*_yearly_stats.csv`
- `*_topic_info.csv`
- `*_documents_topics.csv`

## Added Columns

`*_documents_topics.csv` includes:

- `topic`
- `topic_probability_max`

`topic_probability_max` is populated for both BERTopic probability shapes:

- 2D probability matrix: uses row-wise max
- 1D probability vector: uses the value directly

`*_topic_info.csv` includes:

- `topic_label_generated`

`topic_label_generated` is generated from BERTopic's `model.generate_topic_labels(...)` with a fallback to top topic words if needed.

## Notes

- `run_ev_all.py` default source-tag mapping includes `electricvehicles=evforum`.
- If a comments file is missing for a matched submissions file, the scripts continue with an empty comments table for that source.
