# Run the instruction model on the Windows desktop

The machine this note is written for is a desktop RTX 4070 with 16 GB of VRAM, Windows, Anaconda. The cloud VM that prepared the sheet has no GPU, so the 23,000-post run has not been done. The small NLI model remains the baseline already in `outputs/framing_labels.csv`. This run is the open instruction model. It is still not human validation.

## First session: 100 posts

Do this before the full file. The posts are already in git as `sample/smoke_100.csv` (51 enthusiast, 49 shopper). `selftext` is an excerpt of at most 800 characters, with `u/` names removed. It is enough to time the model and to read the JSON. It is not the full post, and it is not a result.

From `v2`, after Ollama is serving:

```
python llm_label.py --backend ollama --model qwen2.5:7b-instruct --base-url http://127.0.0.1:11434/v1 --posts sample/smoke_100.csv --out outputs/llm_smoke.csv --cache data/raw/llm_smoke.jsonl --batch-size 1
```

Checklist, in order:

1. Time the command. Divide the minutes by 100. Multiply by 22,827. That is the planning time for the full file on this machine.
2. Open `outputs/llm_smoke.csv`. `ok` should be true on every row. `n` should be 100. If `ok` is false, the model did not return JSON. Stop.
3. `stage` should be one of `awareness`, `consideration`, `evaluation`, `purchase`, `onboarding`, `long_term_ownership`, `unclear`. Anything else is an unknown stage. The downstream command drops those rows and counts them.
4. `families` and `aspects` are code names separated by `|`. A name that is not in `annotation/CODEBOOK.md` is a bad code.
5. Spot-read 10 rows. The `evidence` quote should be words that appear in that row's title or selftext. The stage should match where the writer is, not the mere presence of a word like "charging." If the quotes are invented, or the stage is a keyword, change the prompt before the long run.
6. Rebuild the tables for this smoke file only, with no bootstrap, so you can see that the pipeline runs:

```
python run_after_llm.py --llm outputs/llm_smoke.csv --out outputs/smoke_tables --n-boot 0
```

Those tables are a smoke test. Do not put them in the paper.

## What to install

Use Anaconda. A 4-bit 7B model fits in 16 GB. A 14B 4-bit model is tight once the prompt and the Windows desktop are using memory. Start with 7B. Try 14B only if `nvidia-smi` shows several gigabytes free after the 7B model is loaded.

Ollama is the straightforward Windows path. vLLM is easier under WSL2 than under native Windows. `transformers` with 4-bit works when bitsandbytes is installed, and it is the slowest of the three for this many posts.

```
ollama pull qwen2.5:7b-instruct
```

Other checkpoints the script accepts, if you prefer them:

- `Qwen/Qwen2.5-7B-Instruct` (recommended)
- `Qwen/Qwen2.5-14B-Instruct` (only if 7B leaves headroom)
- `meta-llama/Llama-3.1-8B-Instruct` (a gated Hugging Face download; Ollama's `llama3.1:8b-instruct-q4_0` is the local equivalent)

## Where the post text is

`outputs/framing_labels.csv` has 22,827 ids and no body text. Full text is not in git. On this VM a fetch was started into gitignored `v2/data/raw/label_text.jsonl` (id, hashed author, time, title, 800-character excerpt). Copy that file to the desktop if you have it, or rebuild it there:

```
cd v2
python fetch_label_text.py
python -c "import pandas as pd, json; rows=[json.loads(l) for l in open('data/raw/label_text.jsonl') if l.strip()]; pd.DataFrame(rows).to_csv('data/raw/label_posts.csv', index=False)"
```

`label_posts.csv` needs columns `id`, `title`, and `excerpt` or `selftext`. The labeler reads `title` plus `selftext` or `text`. If the column is named `excerpt`, rename it to `selftext` before the run. Do not commit `label_posts.csv`.

The fetch is about 100 requests to Arctic Shift. Plan on well under an hour unless the API asks you to slow down.

## The run

From `v2`, with Ollama already serving on port 11434:

```
python llm_label.py --backend ollama --model qwen2.5:7b-instruct --base-url http://127.0.0.1:11434/v1 --posts data/raw/label_posts.csv --out outputs/llm_labels.csv --cache data/raw/llm_labels.jsonl --batch-size 1
```

vLLM under WSL2, after `vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 4096`:

```
python llm_label.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct --base-url http://127.0.0.1:8000/v1 --posts data/raw/label_posts.csv --out outputs/llm_labels.csv --cache data/raw/llm_labels.jsonl
```

transformers 4-bit, if you would rather not run a server:

```
python llm_label.py --backend transformers --model Qwen/Qwen2.5-7B-Instruct --posts data/raw/label_posts.csv --out outputs/llm_labels.csv --cache data/raw/llm_labels.jsonl
```

Add `--no-four-bit` only if you are sure the full weights fit. On 16 GB they do not.

Stop the job whenever you need to. The jsonl cache is keyed by post id, model name, and codebook version. Starting the same command again skips posts already labeled.

Each output row stores `model_name`, `model_version`, and `prompt_hash`. The small NLI rows in `framing_labels.csv` are a different model. Keep both files. A small-versus-large comparison can be a later paper. Do not overwrite `framing_labels.csv` with this run.

`evidence` is a short quote. Review it before committing `llm_labels.csv`. The quotes are unvalidated model output.

After the full file exists:

```
python run_after_llm.py --llm outputs/llm_labels.csv --out outputs/after_llm
```

That rebuilds funnel shares, the stage-by-family heatmap, the discontinuity table (bandwidth grid, donut, placebos, week-cluster interval), the transition rates, and agreement with the small model. Pass `--coder-a` and `--coder-b` only when both sheets have been filled by people. The command does not invent those sheets. Add `--n-boot 200` if you want the interval; the default is 200.

## How long 23,000 posts take

Time the first 100 posts and multiply. Planning numbers for an RTX 4070 16 GB, 4-bit 7B, one post at a time, a codebook prompt of roughly 700 tokens and a JSON answer of roughly 120 tokens:

- Prefill is the slow part if the prompt is not cached. At a few thousand tokens per second of prefill, 100 posts is on the order of 5 to 15 minutes.
- Generation of the JSON is on the order of 1 to 2 seconds per post at typical 7B 4-bit speeds.
- A single stream is then about **6 to 14 hours** for 22,827 posts.
- Ollama or llama.cpp with a cached codebook prompt, or vLLM with several sequences in flight, is the short end of that range, closer to **3 to 6 hours**.
- A 14B model on the same card is slower and may run out of memory. Do not plan on it for the first pass.

If the first 100 posts take longer than 15 minutes, stop and check that the GPU is the device in use (`nvidia-smi`) and that the model is a 4-bit or Q4 quant, not fp16.

## Optional: a rented GPU, if the desktop is busy

Nothing was rented. This is a price check, not a recommendation to spend money. The desktop run is the plan.

On 21 October 2026 the RunPod pricing page (`https://www.runpod.io/pricing`) listed an on-demand RTX 4090 at $0.34 per hour on Community Cloud and $0.89 per hour on Secure Cloud (24 GB, which is enough for a 4-bit 7B model). The same page listed an L4 at $0.44 per hour (Community) and $0.59 per hour (Secure).

At the Secure Cloud 4090 rate, 14 hours is about $12, and a full day left running by mistake is about $21. At the Community rate the same 14 hours is about $5. Even 48 hours at $0.89 is about $43. All of those are under $100. Marketplace prices move, and a failed start still costs the hours the machine was up. The local 4070 costs nothing if it is free.

## What this run is not

It does not replace the 800-post sheet. The labmate and Daisy code `annotation/annotation_sheet.csv` against `annotation/CODEBOOK.md`. `score_agreement.py` compares those sheets with each other and with whichever model key you point it at. Until that happens, every share is unvalidated.
