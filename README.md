# EV Reddit BERTopic Pipeline

This project processes EV-related Reddit data in three stages:

1. Topic extraction from raw Reddit submissions/comments.
2. Rule-based topic refinement.
3. LLM refinement to produce human-readable topic labels and summaries.

Core modules live in `src/ev_bertopic/`.

## 📁 Standard Project Layout

This repo now follows the common Python project structure:

- `src/ev_bertopic/`: package code (pipelines)
- `scripts/`: executable runner scripts
- `data/`: raw and filtered Reddit datasets
- `output/`: extraction/refinement/exploration outputs
- `code/`: notebooks and ad-hoc analysis workspace

Install package in editable mode (recommended):

```bash
pip install -e .
```

## Pipeline Stages

### 1) 📥 Extract Topics From Raw Reddit Data

Use BERTopic extraction runners:

- `scripts/run_ev_reddit_extract.py`
- `scripts/run_ev_other_extract.py`
- `scripts/run_ev_all_extract.py`

`run_ev_all_extract.py` reads from `data/data_all` and writes extraction outputs to `output/topic_extraction` by default.

Main extraction outputs:

- `all_subreddits_yearly_stats.csv`
- `all_subreddits_topic_info.csv`
- `all_subreddits_documents_topics.csv`

### 2) 🧠 Rule-Based Topic Refinement

Rule-based refinement is implemented in `src/ev_bertopic/topic_refine_pipeline.py`.

It consumes extraction topic info and generates cleaned labels/keyword columns.

### 3) 🤖 LLM Topic Refinement

LLM refinement is implemented in `src/ev_bertopic/topic_llm_pipeline.py`.

It consumes the Stage 2 refined CSV and produces final human-readable topic labels and summaries.

## 🚀 End-to-End Refinement Runner

Use `scripts/run_ev_all_refine.py` to chain Stage 2 and Stage 3 for the `run_ev_all_extract.py` extraction output.

It performs:

1. Read extraction output: `all_subreddits_topic_info.csv`.
2. Run rule-based refinement and save intermediate CSV.
3. Run LLM refinement and save final CSV.

Run from the repository root:

```bash
python scripts/run_ev_all_refine.py
```

Example:

```bash
python scripts/run_ev_all_refine.py \
	--extraction-dir output/topic_extraction \
	--refinement-dir output/topic_refinement
```

Default outputs:

- 🟡 Intermediate: `output/topic_refinement/all_subreddits_topic_labels_refined.csv`
- ✅ Final: `output/topic_refinement/all_subreddits_topic_labels_llm.csv`
- 📝 Gemini call log: `output/topic_refinement/gemini_call_log_all_subreddits.json`

## 📊 Explore Pipeline

`src/ev_bertopic/topic_explore_pipeline.py` is intended to read final cleaned topic results directly (preferably LLM outputs) and does not need to run rule-based refinement classes internally.

If `topic_label_llm` is present, exploration uses it as the working topic label.

## ⚡ Quick Start Commands

```bash
# 1) 📥 Extract topics from raw Reddit data
python scripts/run_ev_all_extract.py

# 2) 🧠🤖 Run rule-based + LLM refinement chain (saves intermediate and final outputs)
python scripts/run_ev_all_refine.py
```

## ℹ️ Notes

- `run_ev_all_extract.py` default source-tag mapping includes `electricvehicles=evforum`.
- If a comments file is missing for a matched submissions file, extraction continues with an empty comments table for that source.
