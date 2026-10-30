#!/usr/bin/env python3
"""Local instruction-model labels for stage, family, aspect, and stance.

The small NLI model in ``ev_v2/annotate.py`` stays the baseline. This script
is the open instruction model Daisy will run on her own GPU. It does not
call a paid API. On a machine with no GPU, use ``--backend mock``.

Each output row records the model name and version. A jsonl cache lets a
stopped run resume. The prompt asks for JSON only.

From ``v2/``::

    python llm_label.py --backend mock --posts tests/fixtures/llm_posts.csv --out /tmp/llm_out.csv
    python llm_label.py --backend ollama --posts data/raw/label_posts.csv --out outputs/llm_labels.csv

Windows steps are in ``RUN_ON_MY_GPU.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ev_v2.taxonomy import (  # noqa: E402
    ASPECT_CODES,
    ASPECTS,
    FAMILIES,
    FAMILY_CODES,
    STAGE_CODES,
    STAGES,
    STANCE_CODES,
)

PROMPT_VERSION = "funnel-v2"
# Comments v2 (COMMENTS_V2_PLAN.md): same codebook, parent title and excerpt in the prompt.
PROMPT_VERSION_COMMENTS_V2 = "funnel-v2-comment-parent"
PARENT_EXCERPT_CHARS = 300
PARENT_CONTEXT_NOTE = (
    "The subreddit, parent title, and parent excerpt are context only. Code the comment writer's own "
    "journey position, topics, and stance, not the parent post's."
)
# Largest open instruct model Daisy chose for a 16 GB card. Q4_K_M is about 9 GB.
# The 7B tag is the fallback when 14B is too slow or runs out of memory.
DEFAULT_OLLAMA_MODEL = "qwen2.5:14b-instruct"
FALLBACK_OLLAMA_MODEL = "qwen2.5:7b-instruct"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-14B-Instruct"
FALLBACK_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_NUM_CTX = 4096
# Daisy's run of record, 22 October 2026: 14B on the submissions plus batch 1.
N_SUBMISSIONS = 22827
N_COMMENT_BATCH1 = 26088
PLAN_HOURS_14B_WITH_BATCH1 = 60.0


class LabelingStopped(RuntimeError):
    """The backend went away. Rows already flushed stay in the cache."""


def smoke_limit_minutes(
    budget_hours: float = PLAN_HOURS_14B_WITH_BATCH1,
    n_items: int = N_SUBMISSIONS + N_COMMENT_BATCH1,
    smoke_n: int = 100,
) -> float:
    """Minutes for ``smoke_n`` items at which the full run would pass ``budget_hours``."""
    if n_items <= 0:
        return 0.0
    return budget_hours * smoke_n / n_items * 60.0


def resolve_model(backend: str, model: str, fallback_7b: bool) -> str:
    """Pick the checkpoint. An explicit --model wins, except the 7B fallback flag."""
    if fallback_7b:
        return FALLBACK_HF_MODEL if backend == "transformers" else FALLBACK_OLLAMA_MODEL
    if model:
        return model
    if backend == "transformers":
        return DEFAULT_HF_MODEL
    if backend in {"ollama", "vllm"}:
        return DEFAULT_OLLAMA_MODEL
    return "mock"


def codebook_prompt(title: str, body: str, parent_context: bool = False) -> str:
    stages = "\n".join(f"- {code}: {label}" for code, label in STAGES)
    families = "\n".join(f"- {code}: {label}" for code, label in FAMILIES)
    aspects = "\n".join(f"- {code}: {label}" for code, label in ASPECTS)
    stances = ", ".join(STANCE_CODES)
    text = f"{title}\n{body}".strip()
    if len(text) > 1500:
        text = text[:1500]
    kind = "comment" if text.lstrip().lower().startswith("[comment]") else "submission"
    context = f"{PARENT_CONTEXT_NOTE}\n" if parent_context else ""
    return f"""You code one Reddit {kind} about electric vehicles.
{context}Assign the stage from where the writer is in the purchase journey, not from a keyword alone. Charging can be evaluation or onboarding depending on whether they already own the car.
Assign every topic family that is a real subject of the post, and every finer aspect that is present.
Stance is the writer's position on electric vehicles as a class, not on one broken charger.

Stages (choose one, or unclear):
{stages}
- unclear: the post does not show a journey position

Families (zero or more):
{families}

Aspects (zero or more):
{aspects}

Stance (choose one): {stances}

Use only the codes listed above, spelled exactly as listed. Do not invent new codes. If no aspect applies, use an empty list.

Return one JSON object and nothing else, with keys stage, families, aspects, stance, evidence.
evidence is at most 20 words copied from the text. Use an empty string if you cannot quote it.

Text:
{text}
"""


def prompt_hash(version: str = PROMPT_VERSION, parent_context: bool = False) -> str:
    sample = codebook_prompt("title", "body", parent_context=parent_context)
    # Hash the instructions, not the post, so one codebook version shares a cache.
    head = sample.split("Submission:", 1)[0]
    return hashlib.sha256(f"{version}|{head}".encode()).hexdigest()[:16]


def _clean_text(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    text = str(value)
    if text.strip().lower() in {"nan", "none", "[removed]", "[deleted]"}:
        return ""
    return text.strip()


def comment_v2_text(subreddit, parent_title, parent_excerpt, body) -> str:
    """Comment text with its parent post as context. Missing parent fields stay empty, never invented."""
    sub = _clean_text(subreddit)
    excerpt = _clean_text(parent_excerpt)[:PARENT_EXCERPT_CHARS]
    lines = [
        "[comment]",
        f"Subreddit: r/{sub}" if sub else "Subreddit:",
        f"Parent title: {_clean_text(parent_title)}",
        f"Parent excerpt: {excerpt}",
        "Comment:",
        _clean_text(body),
    ]
    return "\n".join(lines)


def parse_label(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return {"ok": False, "error": "no_json"}
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {"ok": False, "error": "bad_json"}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "not_object"}
    families = payload.get("families") or []
    aspects = payload.get("aspects") or []
    if isinstance(families, str):
        families = [families]
    if isinstance(aspects, str):
        aspects = [aspects]
    # Codes outside the codebook never pass silently: drop them from the label and record them.
    families = [str(item) for item in families]
    aspects = [str(item) for item in aspects]
    stage = str(payload.get("stage") or "unclear")
    # An unknown stage is kept as returned (run_after_llm drops and counts it) and flagged here.
    unknown = [f"stage:{stage}"] if stage not in STAGE_CODES else []
    unknown += [f"family:{c}" for c in families if c not in FAMILY_CODES]
    unknown += [f"aspect:{c}" for c in aspects if c not in ASPECT_CODES]
    families = [c for c in families if c in FAMILY_CODES]
    aspects = [c for c in aspects if c in ASPECT_CODES]
    evidence = str(payload.get("evidence") or "")
    words = evidence.split()
    if len(words) > 20:
        evidence = " ".join(words[:20])
    return {
        "ok": True,
        "stage": stage,
        "families": families,
        "aspects": aspects,
        "stance": str(payload.get("stance") or "no_stance"),
        "evidence": evidence,
        "unknown_codes": unknown,
    }


class MockBackend:
    """Fixed JSON, so the pipeline can be tested with no model download."""

    name = "mock"
    version = "mock-0"

    def complete(self, prompts: list[str]) -> list[str]:
        body = {
            "stage": "evaluation",
            "families": ["charging_energy"],
            "aspects": ["charging_access"],
            "stance": "mixed",
            "evidence": "mock evidence",
        }
        return [json.dumps(body) for _ in prompts]


class OpenAICompatBackend:
    """vLLM uses chat-completions. Ollama uses its native chat call so num_ctx is applied."""

    def __init__(self, base_url: str, model: str, version: str | None = None, num_ctx: int | None = None, native_ollama: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = model
        self.version = version or model
        self.num_ctx = num_ctx
        self.native_ollama = native_ollama

    def _request(self, prompt: str) -> str:
        if self.native_ollama:
            root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
            url = root + "/api/chat"
            payload_body = {
                "model": self.model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0, "num_ctx": self.num_ctx or DEFAULT_NUM_CTX},
                "keep_alive": "60m",  # keep the model in VRAM between posts and sessions
            }
        else:
            url = self.base_url + "/chat/completions"
            payload_body = {
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload_body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if self.native_ollama:
            return payload["message"]["content"]
        return payload["choices"][0]["message"]["content"]

    def complete(self, prompts: list[str]) -> list[str]:
        outputs = []
        for prompt in prompts:
            last: Exception | None = None
            for attempt in range(3):
                try:
                    outputs.append(self._request(prompt))
                    last = None
                    break
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                    last = exc
                    time.sleep(2 * (attempt + 1))
            if last is not None:
                raise LabelingStopped(f"backend unavailable after 3 tries ({last})")
        return outputs


class TransformersBackend:
    """4-bit Hugging Face generate. Imports torch only when this backend is used."""

    def __init__(self, model: str, four_bit: bool = True):
        self.model_name = model
        self.four_bit = four_bit
        self.name = model
        self._pipe = None
        self.version = model

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"device_map": "auto"}
        if self.four_bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.version = getattr(model.config, "_commit_hash", None) or self.model_name
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._pipe = True

    def complete(self, prompts: list[str]) -> list[str]:
        self._load()
        outputs = []
        for prompt in prompts:
            tokens = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            tokens = {key: value.to(self._model.device) for key, value in tokens.items()}
            with self._torch.no_grad():
                generated = self._model.generate(**tokens, max_new_tokens=200, do_sample=False)
            new_tokens = generated[0][tokens["input_ids"].shape[1] :]
            outputs.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True))
        return outputs


def _load_cache(path: Path) -> dict[tuple[str, str, str], dict]:
    """Last complete row wins. A torn last line is cut off so the next append cannot glue to it."""
    done: dict[tuple[str, str, str], dict] = {}
    if not path.exists() or path.stat().st_size == 0:
        return done
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        newline = raw.rfind(b"\n")
        tail = raw[newline + 1 :]
        try:
            json.loads(tail.decode("utf-8"))
            raw = raw + b"\n"
        except (UnicodeDecodeError, json.JSONDecodeError):
            raw = raw[: newline + 1] if newline >= 0 else b""
        path.write_bytes(raw)
    if not raw:
        return done
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or "id" not in row:
            continue
        done[(str(row["id"]), str(row.get("model_name") or ""), str(row.get("prompt_hash") or ""))] = row
    return done


def _eta(seconds: float) -> str:
    if seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "unknown"
    whole = int(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def _row_text(row) -> str:
    body = getattr(row, "selftext", None)
    if body is None or (isinstance(body, float) and body != body) or str(body) in {"", "nan"}:
        body = getattr(row, "body", None) or getattr(row, "text", None) or getattr(row, "excerpt", "") or ""
    return str(body)


def label_posts(frame, backend, cache_path: Path, batch_size: int = 4, max_new: int | None = None) -> "object":
    import pandas as pd

    parent_mode = "parent_title" in getattr(frame, "columns", [])
    version = PROMPT_VERSION_COMMENTS_V2 if parent_mode else PROMPT_VERSION
    digest = prompt_hash(version, parent_context=parent_mode)
    cache = _load_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pending = []
    skipped = 0
    seen: set[str] = set()
    for row in frame.itertuples(index=False):
        doc_id = str(row.id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        key = (doc_id, backend.name, digest)
        if key in cache:
            skipped += 1
            continue
        title = str(getattr(row, "title", "") or "")
        kind = getattr(row, "text_type", "")
        if kind is None or (isinstance(kind, float) and kind != kind) or str(kind) in {"", "nan"}:
            kind = "submission"
        else:
            kind = str(kind)
        if parent_mode and kind == "comment":
            text = comment_v2_text(
                getattr(row, "subreddit", ""),
                getattr(row, "parent_title", ""),
                getattr(row, "parent_excerpt", ""),
                _row_text(row),
            )
            pending.append((doc_id, codebook_prompt("", text, parent_context=True), kind))
            continue
        prefix = "[comment]\n" if kind == "comment" else ""
        pending.append((doc_id, codebook_prompt(title, prefix + _row_text(row)), kind))
    file_total = skipped + len(pending)
    if max_new is not None:
        pending = pending[: max(0, max_new)]
    print(
        f"resume: {skipped} already labeled, {len(pending)} this session, {file_total} ids in this file",
        flush=True,
    )
    started = time.monotonic()
    new_count = 0
    handle = cache_path.open("a", encoding="utf-8")
    try:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            raws = backend.complete([prompt for _, prompt, _kind in chunk])
            for (doc_id, _, kind), raw in zip(chunk, raws):
                parsed = parse_label(raw)
                stored = {
                    "id": doc_id,
                    "text_type": kind,
                    "model_name": backend.name,
                    "model_version": backend.version,
                    "prompt_hash": digest,
                    "prompt_version": version,
                    "stage": parsed.get("stage", ""),
                    "families": "|".join(parsed.get("families") or []),
                    "aspects": "|".join(parsed.get("aspects") or []),
                    "stance": parsed.get("stance", ""),
                    "evidence": parsed.get("evidence", ""),
                    "unknown_codes": "|".join(parsed.get("unknown_codes") or []),
                    "ok": bool(parsed.get("ok")),
                    "validated": False,
                }
                cache[(doc_id, backend.name, digest)] = stored
                handle.write(json.dumps(stored) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                new_count += 1
                elapsed = time.monotonic() - started
                rate = new_count / elapsed if elapsed else 0.0
                remaining = file_total - skipped - new_count
                eta = remaining / rate if rate else float("inf")
                if stored["unknown_codes"]:
                    print(f"unknown codes flagged for {doc_id}: {stored['unknown_codes']}", flush=True)
                print(
                    f"progress {skipped + new_count}/{file_total} "
                    f"new={new_count} elapsed={_eta(elapsed)} eta={_eta(eta)}",
                    flush=True,
                )
    finally:
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
    rows = []
    seen = set()
    for row in frame.itertuples(index=False):
        doc_id = str(row.id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        key = (doc_id, backend.name, digest)
        if key in cache:
            rows.append(cache[key])
    if max_new is not None and new_count >= max_new and skipped + new_count < file_total:
        print(f"session cap {max_new} reached. Run the same command again to continue.", flush=True)
    return pd.DataFrame(rows)


def _label_signature(row: dict) -> str:
    keep = {
        "stage": row.get("stage"),
        "families": row.get("families"),
        "aspects": row.get("aspects"),
        "stance": row.get("stance"),
        "evidence": row.get("evidence"),
        "model_name": row.get("model_name"),
        "prompt_hash": row.get("prompt_hash"),
        "text_type": row.get("text_type"),
        "ok": row.get("ok"),
    }
    return json.dumps(keep, sort_keys=True, default=str)


def merge_label_files(expect_csvs: list[Path], cache_jsonls: list[Path], out_csv: Path) -> dict:
    """One row per expected id, valid JSON, no conflicting duplicates. Writes the csv only when that holds."""
    import pandas as pd

    expected: list[str] = []
    repeated: list[str] = []
    seen_expect: set[str] = set()
    for path in expect_csvs:
        frame = pd.read_csv(path)
        if "id" not in frame.columns:
            return {"ok": False, "error": f"{path} has no id column", "missing": [], "extra": [], "conflicts": [], "invalid": []}
        for doc_id in frame["id"].astype(str):
            if doc_id in seen_expect:
                repeated.append(doc_id)
            seen_expect.add(doc_id)
            expected.append(doc_id)
    rows_by_id: dict[str, dict] = {}
    conflicts: list[str] = []
    invalid: list[str] = []
    for path in cache_jsonls:
        if not path.exists():
            invalid.append(f"missing cache {path}")
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid.append(f"{path.name}:{line_no} not json")
                continue
            if not isinstance(row, dict) or "id" not in row:
                invalid.append(f"{path.name}:{line_no} not an object with id")
                continue
            if not bool(row.get("ok")):
                invalid.append(f"{path.name}:{line_no} id {row.get('id')} ok is not true")
                continue
            doc_id = str(row["id"])
            signature = _label_signature(row)
            prior = rows_by_id.get(doc_id)
            if prior is None:
                row = dict(row)
                row["_signature"] = signature
                rows_by_id[doc_id] = row
            elif prior["_signature"] != signature:
                conflicts.append(doc_id)
    missing = [doc_id for doc_id in seen_expect if doc_id not in rows_by_id]
    extra = sorted(set(rows_by_id) - seen_expect)
    ok = not repeated and not invalid and not conflicts and not missing and not extra
    report = {
        "ok": ok,
        "n_expected": len(seen_expect),
        "n_labeled": len(rows_by_id),
        "missing": missing[:20],
        "n_missing": len(missing),
        "extra": extra[:20],
        "n_extra": len(extra),
        "conflicts": sorted(set(conflicts))[:20],
        "n_conflicts": len(set(conflicts)),
        "invalid": invalid[:20],
        "n_invalid": len(invalid),
        "n_repeated_expect": len(repeated),
    }
    if ok:
        ordered = []
        written: set[str] = set()
        for doc_id in expected:
            if doc_id in rows_by_id and doc_id not in written:
                row = {key: value for key, value in rows_by_id[doc_id].items() if key != "_signature"}
                ordered.append(row)
                written.add(doc_id)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ordered).to_csv(out_csv, index=False)
        report["wrote"] = str(out_csv)
        report["n_wrote"] = len(ordered)
    return report


def _print_merge_report(report: dict) -> None:
    if report.get("error"):
        print(report["error"], flush=True)
        return
    status = "ok" if report["ok"] else "failed"
    print(
        f"merge {status}: expected {report['n_expected']} labeled {report['n_labeled']} "
        f"missing {report['n_missing']} extra {report['n_extra']} "
        f"conflicts {report['n_conflicts']} invalid {report['n_invalid']} "
        f"repeated_ids {report['n_repeated_expect']}",
        flush=True,
    )
    for key in ("missing", "extra", "conflicts", "invalid"):
        sample = report.get(key) or []
        if sample:
            print(f"  {key}: {sample}", flush=True)
    if report.get("wrote"):
        print(f"wrote {report['wrote']} rows={report['n_wrote']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local 14B labels for the submissions plus comment batch 1. 7B is a fallback, not the run."
    )
    parser.add_argument("--backend", choices=("mock", "ollama", "vllm", "transformers"), default="mock")
    parser.add_argument("--model", default="", help="Empty uses qwen2.5:14b-instruct, the run of record.")
    parser.add_argument(
        "--fallback-7b",
        action="store_true",
        help="Fallback only: use qwen2.5:7b-instruct instead of the 14B run.",
    )
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--posts", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--caches", action="append", default=None, type=Path, help="Repeat for each jsonl when merging.")
    parser.add_argument("--expect", action="append", default=None, type=Path, help="Repeat for each id file when merging.")
    parser.add_argument("--merge-validate", action="store_true", help="Check every expected id once, then write --out.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=None, help="Stop after this many new rows so a session can end cleanly.")
    parser.add_argument("--four-bit", action="store_true", default=True)
    parser.add_argument("--no-four-bit", action="store_true")
    args = parser.parse_args(argv)
    if args.merge_validate:
        if not args.expect or not args.caches or args.out is None:
            parser.error("--merge-validate needs at least one --expect, one --caches, and --out")
        report = merge_label_files(args.expect, args.caches, args.out)
        _print_merge_report(report)
        return 0 if report["ok"] else 1
    if args.posts is None or args.out is None:
        parser.error("--posts and --out are required")
    import pandas as pd

    posts = pd.read_csv(args.posts)
    model = resolve_model(args.backend, args.model, args.fallback_7b)
    if args.backend == "mock":
        backend = MockBackend()
    elif args.backend in {"ollama", "vllm"}:
        base = args.base_url or ("http://127.0.0.1:11434/v1" if args.backend == "ollama" else "http://127.0.0.1:8000/v1")
        backend = OpenAICompatBackend(base, model, num_ctx=args.num_ctx, native_ollama=args.backend == "ollama")
    else:
        backend = TransformersBackend(model, four_bit=not args.no_four_bit)
    cache = args.cache or args.out.with_suffix(".jsonl")
    try:
        labeled = label_posts(posts, backend, cache, batch_size=max(1, args.batch_size), max_new=args.max_new)
    except LabelingStopped as exc:
        print(f"stopped ({exc}). Run the same command again. Finished ids are in {cache}.", flush=True)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(args.out, index=False)
    print(f"wrote {args.out} rows={len(labeled)} model={backend.name} version={backend.version}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
