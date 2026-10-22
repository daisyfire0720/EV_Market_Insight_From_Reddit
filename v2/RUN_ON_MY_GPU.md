# Run the instruction model on the Windows desktop

The machine this note is written for is a desktop RTX 4070 with 16 GB of VRAM, Windows, Anaconda. The cloud VM that prepared the sheet has no GPU, so the 23,000-post run has not been done and the 14B model has not been timed here. The small NLI model remains the baseline already in `outputs/framing_labels.csv`. This run is the open instruction model. It is still not human validation.

Daisy's decision on 22 October 2026: the default is the largest open instruct model that fits the 16 GB card, `qwen2.5:14b-instruct` (Ollama, Q4_K_M, about 9 GB of weights). The fallback is `qwen2.5:7b-instruct` if the 14B model is too slow or runs out of memory.

## First session: 100 posts

Do this before the full file. The posts are already in git as `sample/smoke_100.csv` (51 enthusiast, 49 shopper). `selftext` is an excerpt of at most 800 characters, with `u/` names removed. It is enough to time the model and to read the JSON. It is not the full post, and it is not a result.

From `v2`, after Ollama is serving:

```
ollama pull qwen2.5:14b-instruct
python llm_label.py --backend ollama --base-url http://127.0.0.1:11434/v1 --num-ctx 4096 --posts sample/smoke_100.csv --out outputs/llm_smoke.csv --cache data/raw/llm_smoke.jsonl --batch-size 1
```

An empty `--model` is `qwen2.5:14b-instruct`. Pass `--model` only if you want a different tag. Pass `--fallback-7b` to use `qwen2.5:7b-instruct` instead:

```
ollama pull qwen2.5:7b-instruct
python llm_label.py --backend ollama --fallback-7b --base-url http://127.0.0.1:11434/v1 --num-ctx 4096 --posts sample/smoke_100.csv --out outputs/llm_smoke_7b.csv --cache data/raw/llm_smoke_7b.jsonl --batch-size 1
```

Keep the 14B and 7B caches in different files. The cache key includes the model name.

Checklist, in order:

1. Time the command. Divide the minutes by 100. Multiply by 22,827 for submissions alone. If you are also labeling comments, multiply by `(22827 + N) / 22827`, where `N` is the number of rows in `outputs/comment_meta.csv`.
2. Open `outputs/llm_smoke.csv`. `ok` should be true on every row. `n` should be 100. If `ok` is false, the model did not return JSON. Stop.
3. `stage` should be one of `awareness`, `consideration`, `evaluation`, `purchase`, `onboarding`, `long_term_ownership`, `unclear`. Anything else is an unknown stage. The downstream command drops those rows and counts them.
4. `families` and `aspects` are code names separated by `|`. A name that is not in `annotation/CODEBOOK.md` is a bad code.
5. Spot-read 10 rows. The `evidence` quote should be words that appear in that row's title or selftext. The stage should match where the writer is, not the mere presence of a word like "charging." If the quotes are invented, or the stage is a keyword, change the prompt before the long run.
6. Rebuild the tables for this smoke file only, with no bootstrap, so you can see that the pipeline runs:

```
python run_after_llm.py --llm outputs/llm_smoke.csv --out outputs/smoke_tables --n-boot 0
```

Those tables are a smoke test. Do not put them in the paper. If `outputs/post_scores.csv` is present, the same command also writes the score-at-least-2 sensitivity next to the main tables. The main tables still include every row.

## Context length

`--num-ctx 4096` is the default. The 14B Q4_K_M weights are about 9 GB. A 4096-token context keeps the key-value cache small (well under 1 GB with grouped-query attention on this model). Do not raise the context to 32k. That is what runs a 16 GB card out of memory once Windows is also using the GPU. The codebook prompt is about 700 tokens and the JSON answer is about 120, so 4096 is enough for one post.

If `nvidia-smi` shows the 14B model plus the cache leaving less than about 1 GB free, stop and rerun the same command with `--fallback-7b`.

## What to install

Use Anaconda. Ollama is the straightforward Windows path. vLLM is easier under WSL2 than under native Windows. `transformers` with 4-bit works when bitsandbytes is installed, and it is the slowest of the three for this many posts.

```
ollama pull qwen2.5:14b-instruct
```

The script's defaults:

- Ollama and vLLM, empty `--model`: `qwen2.5:14b-instruct`
- transformers, empty `--model`: `Qwen/Qwen2.5-14B-Instruct`
- `--fallback-7b`: `qwen2.5:7b-instruct`, or `Qwen/Qwen2.5-7B-Instruct` on the transformers backend

`meta-llama/Llama-3.1-8B-Instruct` still works if you pass it as `--model`. It is a gated Hugging Face download. It is not the default.

## Where the post text is

`outputs/framing_labels.csv` has 22,827 ids and no body text. Full text is not in git. On this VM a fetch was started into gitignored `v2/data/raw/label_text.jsonl` (id, hashed author, time, title, 800-character excerpt). Copy that file to the desktop if you have it, or rebuild it there:

```
cd v2
python fetch_label_text.py
python -c "import pandas as pd, json; rows=[json.loads(l) for l in open('data/raw/label_text.jsonl') if l.strip()]; pd.DataFrame(rows).to_csv('data/raw/label_posts.csv', index=False)"
```

`label_posts.csv` needs columns `id`, `title`, and `excerpt` or `selftext`. The labeler reads `title` plus `selftext`, `body`, or `text`. If the column is named `excerpt`, rename it to `selftext` before the run. Do not commit `label_posts.csv`.

The submission fetch is about 100 requests to Arctic Shift. Plan on well under an hour unless the API asks you to slow down.

## Comments

Comments are a second text type. They are not mixed into the submission rows. `fetch_comments.py` has already pulled comments by the 2,392 hashed repeat authors, and one page of comments on each sampled submission inside the cutoff windows. Repeat-author queries keep the closest 100 comments on each side of a cutoff. Thread queries keep one page of 100. The committed metadata is `outputs/comment_meta.csv` (250,794 comments: hash, time, subreddit, score, link id) and `outputs/comment_cutoff_coverage.csv`. Bodies stay in gitignored `data/raw/comments.jsonl`.

Do not label all 250,794 comments in the same run as the submissions. `outputs/comment_label_plan.csv` is the budget. Cutoff-crossing authors are first (189,179 comments). That is more than the 1,630 comments that still fit with the 22,827 submissions in 30 hours on 14B at the slow planning rate, so the plan is 7B. Label `outputs/comment_label_batch1.csv` (26,088 comments, the closest cutoff-crossing rows). `outputs/comment_label_batch2.csv` (224,706 comments) is optional and is not part of the 30-hour plan.

Fill bodies for batch 1 into the gitignored cache, then build the local label file. Neither file is committed:

```
python fetch_comments.py --backfill outputs/comment_label_batch1.csv
python -c "from ev_v2.comment_batches import materialize_label_input; materialize_label_input('outputs/comment_label_batch1.csv', 'data/raw/comments.jsonl', 'data/raw/comment_label_batch1_posts.csv')"
```

A comment row needs `text_type` equal to `comment` and the text in `body` or `selftext`. The labeler prefixes the body with `[comment]` so the prompt says the item is a comment, and it stores `text_type` on the output row. Submissions stay `text_type=submission`. Label comments into the same `outputs/llm_labels.csv` only after the submission run, or into a separate file and concatenate. Do not overwrite a finished submission file with a comment-only run. The batch csv has no body. The materialized csv does, and it is gitignored.

The submissions-only stage-transition count is the robustness check. `run_after_llm.py` does not drop comments out of the main tables when they are in the label file. Keep a copy of the submission-only label file if you want that robustness table rebuilt on its own.

## The run

From `v2`, with Ollama already serving on port 11434:

```
python llm_label.py --backend ollama --base-url http://127.0.0.1:11434/v1 --num-ctx 4096 --posts data/raw/label_posts.csv --out outputs/llm_labels.csv --cache data/raw/llm_labels.jsonl --batch-size 1
```

If the 14B model is too slow or runs out of memory:

```
python llm_label.py --backend ollama --fallback-7b --base-url http://127.0.0.1:11434/v1 --num-ctx 4096 --posts data/raw/label_posts.csv --out outputs/llm_labels.csv --cache data/raw/llm_labels_7b.jsonl --batch-size 1
```

vLLM under WSL2, after `vllm serve Qwen/Qwen2.5-14B-Instruct --max-model-len 4096`:

```
python llm_label.py --backend vllm --model Qwen/Qwen2.5-14B-Instruct --base-url http://127.0.0.1:8000/v1 --num-ctx 4096 --posts data/raw/label_posts.csv --out outputs/llm_labels.csv --cache data/raw/llm_labels.jsonl
```

transformers 4-bit, if you would rather not run a server (the empty `--model` is the 14B Hugging Face id):

```
python llm_label.py --backend transformers --posts data/raw/label_posts.csv --out outputs/llm_labels.csv --cache data/raw/llm_labels.jsonl
```

Add `--no-four-bit` only if you are sure the full weights fit. On 16 GB they do not.

Stop the job whenever you need to. The jsonl cache is keyed by post id, model name, and codebook version. Starting the same command again skips posts already labeled.

Each output row stores `model_name`, `model_version`, and `prompt_hash`. The small NLI rows in `framing_labels.csv` are a different model. Keep both files. A small-versus-large comparison can be a later paper. Do not overwrite `framing_labels.csv` with this run.

`evidence` is a short quote. Review it before committing `llm_labels.csv`. The quotes are unvalidated model output.

After the full file exists:

```
python run_after_llm.py --llm outputs/llm_labels.csv --out outputs/after_llm
```

That rebuilds funnel shares, the stage-by-family heatmap, the discontinuity table (bandwidth grid, donut, placebos, week-cluster interval), the transition rates, and agreement with the small model. It also writes the score-at-least-2 sensitivity when `outputs/post_scores.csv` is present. Pass `--coder-a` and `--coder-b` only when both sheets have been filled by people. Ryan's sheet is `annotation/coder_ryan.csv`. The command does not invent those sheets. Add `--n-boot 200` if you want the interval; the default is 200.

## How long it takes

This VM has no GPU. Nothing below is a measured tokens-per-second figure for the 14B model. Time the first 100 posts and multiply.

Planning numbers for an RTX 4070 16 GB, one post at a time, a codebook prompt of roughly 700 tokens and a JSON answer of roughly 120 tokens:

- **7B, Q4.** Prefill of 100 posts is on the order of 5 to 15 minutes if the prompt is not cached. Generation is on the order of 1 to 2 seconds per post. A single stream is then about **6 to 14 hours** for 22,827 submissions. A cached codebook prompt, or several sequences in flight, is the short end, closer to **3 to 6 hours**.
- **14B, Q4_K_M.** About twice as slow as the 7B on the same card, as a planning assumption, not a measurement. That puts a single stream near **12 to 28 hours** for the 22,827 submissions, before comments. The slow end of that band is next to 30 hours. Comments make it longer. See the rule below.

**The 30-hour line.** 30 hours times 100 / 22,827 is about **8 minutes**. If the 100-post 14B smoke test takes longer than about 8 minutes, the full 22,827-submission run at that speed passes about 30 hours. Stop the 14B plan for the full file. Use `--fallback-7b`, or the rented-GPU option in the next section. Do not start the full 14B file until that smoke time is known.

Submissions alone are 28 hours on 14B and 14 hours on 7B at the slow end of those bands. Submissions plus batch 1 (22,827 + 26,088) are **30.0 hours on 7B** and **60.0 hours on 14B** at the same rates. Submissions plus every in-window comment (250,794) are about 168 hours on 7B and about 336 hours on 14B. Batch 2 is that overflow. These are planning rates, not a timing from this VM.

**The batch-1 smoke line.** 30 hours times 100 / (22,827 + 26,088) is about **3.7 minutes**. If the 100-post 7B smoke test takes longer than about 3.7 minutes, submissions plus batch 1 pass about 30 hours. Stop and use the rented-GPU option below. The 8-minute line above is only the 14B submission file, before comments.

If the first 100 posts take longer than 15 minutes even on 7B, stop and check that the GPU is the device in use (`nvidia-smi`) and that the model is a 4-bit or Q4 quant, not fp16.

## Optional: a rented GPU, if 14B would pass 30 hours

Nothing was rented. This is a price check, not a recommendation to spend money.

Use this when the 14B smoke test says the 22,827 submissions, or the submissions plus comments, would run past about 30 hours on the 4070. The other option on the same desktop is `--fallback-7b`.

On 21 October 2026 the RunPod pricing page (`https://www.runpod.io/pricing`) listed an on-demand RTX 4090 at $0.34 per hour on Community Cloud and $0.89 per hour on Secure Cloud (24 GB). The same page listed an L4 at $0.44 per hour (Community) and $0.59 per hour (Secure). A 24 GB card has room for the 14B Q4 weights (about 9 GB) and the 4096-token context. Prices were not re-checked after 21 October 2026.

At the Secure Cloud 4090 rate, 14 hours is about $12, a full day is about $21, and 48 hours is about $43. At the Community rate the same 14 hours is about $5. All of those are under $100. Marketplace prices move, and a failed start still costs the hours the machine was up. The local 4070 costs nothing if the 7B fallback is fast enough.

## What this run is not

It does not replace the 800-post sheet. Ryan (coder B) and Daisy (coder A) code `annotation/annotation_sheet.csv` against `annotation/CODEBOOK.md`. Ryan's steps are `annotation/README_FOR_RYAN.md`. `score_agreement.py` compares those sheets with each other and with whichever model key you point it at. Until that happens, every share is unvalidated.
