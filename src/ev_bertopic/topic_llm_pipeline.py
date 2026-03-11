"""LLM topic label refinement pipeline for BERTopic outputs."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


@dataclass
class TopicLLMConfig:
    input_path: str = "../../output/topic_refinement/topic_labels_refined.csv"
    output_path: str = "../../output/topic_refinement/topic_labels_llm.csv"
    embed_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    gemini_model_name: str = "gemini-2.5-flash"
    gemini_api_key: Optional[str] = None
    call_interval_seconds: float = 30.0
    daily_call_limit: int = 500
    call_log_path: str = "../../output/topic_refinement/gemini_call_log.json"
    max_doc_char: int = 1200
    embed_batch_size: int = 128
    sleep_seconds: float = 0.2
    show_progress_bar: bool = True
    temperature: float = 0.0


SYSTEM_PROMPT = """
You analyze Reddit discussions about electric vehicles.

Your job:
Generate a concise human-readable topic label.

Rules:
- 3-8 words preferred
- clear and descriptive
- avoid generic words like "discussion"
- focus on the real theme

Return JSON only:

{
 "topic_label_llm": "...",
 "topic_summary_llm": "..."
}
""".strip()


USER_PROMPT_TEMPLATE = """
BERT Generated Label:
{generated}

Rule-refined Label:
{rule}

Keywords:
{keywords}

KeyBERT Keywords:
{keybert}

Representative Post A:
{docA}

Representative Post B:
{docB}
""".strip()


class TopicLLMPipeline:
    def __init__(self, cfg: Optional[TopicLLMConfig] = None):
        self.cfg = cfg or TopicLLMConfig()
        self._embedder: Optional[SentenceTransformer] = None
        self._client: Any = None
        self._gemini_backend: Optional[str] = None
        self._genai_types: Any = None
        self._last_call_ts: Optional[float] = None

    def _load_call_timestamps(self) -> List[float]:
        path = Path(self.cfg.call_log_path)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [float(x) for x in data]
        except Exception:
            pass
        return []

    def _save_call_timestamps(self, timestamps: Sequence[float]) -> None:
        path = Path(self.cfg.call_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(timestamps)), encoding="utf-8")

    def _enforce_rate_limit(self) -> None:
        now = time.time()
        if self._last_call_ts is not None and self.cfg.call_interval_seconds > 0:
            elapsed = now - self._last_call_ts
            wait_seconds = self.cfg.call_interval_seconds - elapsed
            if wait_seconds > 0:
                print(f"Sleeping {wait_seconds:.1f}s to respect per-call limit...")
                time.sleep(wait_seconds)
                now = time.time()

        cutoff = now - 24 * 60 * 60
        timestamps = sorted(ts for ts in self._load_call_timestamps() if ts >= cutoff)

        if len(timestamps) >= self.cfg.daily_call_limit:
            next_allowed_at = timestamps[0] + 24 * 60 * 60
            wait_for = max(1, int(next_allowed_at - now))
            raise RuntimeError(
                "Gemini 24-hour call limit reached "
                f"({self.cfg.daily_call_limit}). Try again in about {wait_for} seconds."
            )

        timestamps.append(now)
        self._save_call_timestamps(timestamps)
        self._last_call_ts = now

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.cfg.embed_model_name)
        return self._embedder

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                genai_mod = importlib.import_module("google.genai")
                self._genai_types = importlib.import_module("google.genai.types")
                client_kwargs: Dict[str, Any] = {}
                if self.cfg.gemini_api_key:
                    client_kwargs["api_key"] = self.cfg.gemini_api_key
                self._client = genai_mod.Client(**client_kwargs)
                self._gemini_backend = "google-genai"
            except Exception:
                try:
                    # Uses GOOGLE_API_KEY from environment if present.
                    legacy_genai = importlib.import_module("google.generativeai")
                    if self.cfg.gemini_api_key:
                        legacy_genai.configure(api_key=self.cfg.gemini_api_key)
                    self._client = legacy_genai
                    self._gemini_backend = "google-generativeai"
                except Exception as exc:
                    raise ImportError(
                        "Gemini SDK not found. Install either `google-genai` or `google-generativeai`."
                    ) from exc
        return self._client

    @staticmethod
    def _parse_docs(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(x) for x in value]
        if value is None:
            return []
        try:
            if pd.isna(value):
                return []
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(str(value))
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [str(value)]

    def _truncate(self, text: str) -> str:
        return text[: self.cfg.max_doc_char] if len(text) > self.cfg.max_doc_char else text

    def compute_embeddings(self, docs: Sequence[str]) -> np.ndarray:
        if not docs:
            return np.empty((0, 0), dtype=np.float32)
        embedder = self._get_embedder()
        embeddings = embedder.encode(
            list(docs),
            batch_size=self.cfg.embed_batch_size,
            show_progress_bar=self.cfg.show_progress_bar,
            convert_to_numpy=True,
        )
        return embeddings

    @staticmethod
    def centroid_farthest(docs: Sequence[str], embeddings: np.ndarray) -> Tuple[str, str]:
        if len(docs) == 0:
            return "", ""
        if len(docs) == 1:
            return docs[0], docs[0]
        centroid = np.mean(embeddings, axis=0)
        sims = cosine_similarity([centroid], embeddings)[0]
        dists = 1 - sims
        centroid_idx = int(np.argmin(dists))
        farthest_idx = int(np.argmax(dists))
        return docs[centroid_idx], docs[farthest_idx]

    def call_gemini(self, prompt: str) -> Dict[str, Any]:
        client = self._get_client()
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

        if self._gemini_backend == "google-genai":
            resp = client.models.generate_content(
                model=self.cfg.gemini_model_name,
                contents=full_prompt,
                config=self._genai_types.GenerateContentConfig(
                    temperature=self.cfg.temperature,
                    response_mime_type="application/json",
                ),
            )
            text = resp.text or ""
        elif self._gemini_backend == "google-generativeai":
            model = client.GenerativeModel(self.cfg.gemini_model_name)
            resp = model.generate_content(
                full_prompt,
                generation_config={"temperature": self.cfg.temperature},
            )
            text = getattr(resp, "text", "") or ""
        else:
            raise RuntimeError("Gemini backend is not initialized.")

        try:
            return json.loads(text)
        except Exception:
            return {
                "topic_label_llm": text,
                "topic_summary_llm": "",
            }

    def _collect_topic_docs(self, df: pd.DataFrame) -> Tuple[List[List[str]], List[str]]:
        topic_docs: List[List[str]] = []
        all_docs: List[str] = []

        for docs_raw in df["Representative_Docs"]:
            docs = self._parse_docs(docs_raw)
            docs = [self._truncate(str(d)) for d in docs]
            topic_docs.append(docs)
            all_docs.extend(docs)

        return topic_docs, all_docs

    def run(self, input_file: Optional[str] = None, output_file: Optional[str] = None) -> pd.DataFrame:
        input_path = input_file or self.cfg.input_path
        output_path = output_file or self.cfg.output_path

        df = pd.read_csv(input_path)

        print("Collecting representative docs...")
        topic_docs, all_docs = self._collect_topic_docs(df)

        print("Computing embeddings (batched)...")
        embeddings = self.compute_embeddings(all_docs)

        emb_map: Dict[int, np.ndarray] = {}
        idx = 0
        for docs in topic_docs:
            emb_map[id(docs)] = embeddings[idx : idx + len(docs)]
            idx += len(docs)

        results = []
        print("Generating topic labels via Gemini...")

        for i, row in tqdm(df.iterrows(), total=len(df)):
            docs = topic_docs[i]
            doc_emb = emb_map[id(docs)]
            doc_a, doc_b = self.centroid_farthest(docs, doc_emb)

            prompt = USER_PROMPT_TEMPLATE.format(
                generated=row.get("topic_label_bert", ""),
                rule=row.get("topic_label_refined", ""),
                keywords=row.get("topic_keywords_clean", ""),
                keybert=row.get("topic_keybert_clean", ""),
                docA=doc_a,
                docB=doc_b,
            )
            self._enforce_rate_limit()
            out = self.call_gemini(prompt)

            results.append(
                {
                    "topic_id": row.get("Topic", i),
                    "topic_label_bert": row.get("topic_label_bert", ""),
                    "topic_label_refined": row.get("topic_label_refined", ""),
                    "rep_doc_centroid": doc_a,
                    "rep_doc_farthest": doc_b,
                    "topic_label_llm": out.get("topic_label_llm", ""),
                    "topic_summary_llm": out.get("topic_summary_llm", ""),
                }
            )

        out_df = pd.DataFrame(results)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)

        print("LLM Refinement to Human-Readable Topics Done")
        print("File Saved to:", output_path)
        return out_df


def run_pipeline(input_file: str, output_file: str, gemini_api_key: Optional[str] = None) -> pd.DataFrame:
    """Backward-compatible function wrapper around TopicLLMPipeline."""
    cfg = TopicLLMConfig(
        input_path=input_file,
        output_path=output_file,
        gemini_api_key=gemini_api_key,
    )
    return TopicLLMPipeline(cfg).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=TopicLLMConfig.input_path)
    parser.add_argument("--output", default=TopicLLMConfig.output_path)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--call-interval-seconds", type=float, default=TopicLLMConfig.call_interval_seconds)
    parser.add_argument("--daily-call-limit", type=int, default=TopicLLMConfig.daily_call_limit)
    parser.add_argument("--call-log-path", default=TopicLLMConfig.call_log_path)
    args = parser.parse_args()

    cfg = TopicLLMConfig(
        input_path=args.input,
        output_path=args.output,
        gemini_api_key=args.api_key,
        call_interval_seconds=args.call_interval_seconds,
        daily_call_limit=args.daily_call_limit,
        call_log_path=args.call_log_path,
    )
    TopicLLMPipeline(cfg).run()