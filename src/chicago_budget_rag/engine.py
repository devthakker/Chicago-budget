from __future__ import annotations

import json
import math
import os
import re
import subprocess
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[a-zA-Z0-9$%-]+")
HEADING_RE = re.compile(r"^[A-Z0-9\s\-&,./()]{4,}$")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    page_start: int
    page_end: int
    section: str | None
    token_count: int
    toc_like: bool


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    text: str
    source_file: str
    page_start: int
    page_end: int
    section: str | None
    toc_like: bool


class RAGEngine:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_file = self.index_dir / "index.json"
        self.index: dict[str, Any] | None = None

    def build(
        self,
        pdf_paths: list[Path],
        max_tokens: int = 450,
        overlap_tokens: int = 70,
        embedding_model: str | None = None,
        embedding_batch_size: int = 32,
    ) -> dict[str, Any]:
        chunks = self._build_chunks_from_pdfs(pdf_paths, max_tokens, overlap_tokens)

        tokenized: list[list[str]] = [tokenize(c.text) for c in chunks]
        doc_freqs = Counter()
        for toks in tokenized:
            doc_freqs.update(set(toks))

        avg_doc_len = sum(len(toks) for toks in tokenized) / max(len(tokenized), 1)

        chunk_payload = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source_file": c.source_file,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "section": c.section,
                "token_count": c.token_count,
                "toc_like": c.toc_like,
                "term_freq": Counter(tokenized[i]),
                "doc_len": len(tokenized[i]),
            }
            for i, c in enumerate(chunks)
        ]

        embedding_provider = resolve_embedding_provider()
        resolved_embedding_model = embedding_model or default_embedding_model(embedding_provider)

        embeddings: dict[str, list[float]] = {}
        if embedding_provider != "none":
            try:
                texts = [c.text for c in chunks]
                vectors = embed_texts(
                    texts,
                    provider=embedding_provider,
                    model=resolved_embedding_model,
                    batch_size=embedding_batch_size,
                )
                embeddings = {c.chunk_id: v for c, v in zip(chunks, vectors)}
            except Exception as exc:  # pragma: no cover - defensive runtime fallback
                print(f"[build] embeddings disabled due to provider error: {exc}")
                embedding_provider = "none"
                resolved_embedding_model = None

        index = {
            "version": 2,
            "created_from": [str(p.name) for p in pdf_paths],
            "stats": {
                "chunk_count": len(chunks),
                "avg_doc_len": avg_doc_len,
                "chunk_max_tokens": max_tokens,
                "chunk_overlap_tokens": overlap_tokens,
                "embedding_provider": embedding_provider if embeddings else "none",
                "embedding_model": resolved_embedding_model if embeddings else None,
            },
            "bm25": {
                "doc_freqs": dict(doc_freqs),
                "doc_count": len(chunks),
                "avg_doc_len": avg_doc_len,
                "k1": 1.5,
                "b": 0.75,
            },
            "chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "source_file": c["source_file"],
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                    "section": c["section"],
                    "token_count": c["token_count"],
                    "toc_like": c["toc_like"],
                    "doc_len": c["doc_len"],
                    "term_freq": dict(c["term_freq"]),
                }
                for c in chunk_payload
            ],
            "embeddings": embeddings,
        }
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file.write_text(json.dumps(index))
        self.index = index
        return index

    def load(self) -> dict[str, Any]:
        if self.index is None:
            if not self.index_file.exists():
                raise FileNotFoundError(f"Index file not found: {self.index_file}")
            self.index = json.loads(self.index_file.read_text())
        return self.index

    def search(
        self,
        query: str,
        top_k: int = 8,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
    ) -> list[SearchResult]:
        index = self.load()
        chunks = index["chunks"]
        q_tokens = tokenize(query)
        bm25_scores = self._bm25_scores(q_tokens)

        bm25_weight, vector_weight = resolve_retrieval_weights(bm25_weight, vector_weight)
        toc_penalty = _env_float("RAG_TOC_PENALTY", 0.35)

        dense_scores: dict[str, float] = {}
        query_vector: list[float] | None = None
        embeddings: dict[str, list[float]] = index.get("embeddings", {})

        embedding_provider = index.get("stats", {}).get("embedding_provider") or "none"
        embedding_model = index.get("stats", {}).get("embedding_model")

        if embeddings and embedding_provider != "none":
            try:
                query_vector = embed_texts([query], provider=embedding_provider, model=embedding_model, batch_size=1)[0]
            except Exception as exc:  # pragma: no cover - defensive runtime fallback
                print(f"[search] vector query embedding failed; using BM25 only: {exc}")
                query_vector = None

        if query_vector is not None:
            query_norm = _norm(query_vector)
            for chunk_id, vec in embeddings.items():
                dense_scores[chunk_id] = _dot(query_vector, vec) / (query_norm * _norm(vec) + 1e-12)

        bm25_max = max(bm25_scores.values(), default=1.0) or 1.0
        dense_max = max(dense_scores.values(), default=1.0) or 1.0

        blended: list[tuple[str, float]] = []
        for chunk in chunks:
            cid = chunk["chunk_id"]
            bm25_norm = bm25_scores.get(cid, 0.0) / bm25_max
            if dense_scores:
                dense_norm = dense_scores.get(cid, 0.0) / dense_max
                score = bm25_weight * bm25_norm + vector_weight * dense_norm
            else:
                score = bm25_norm

            score += 0.08 * _token_overlap_bonus(query, chunk["text"])
            if chunk.get("toc_like", False):
                score -= toc_penalty
            blended.append((cid, score))

        candidate_multiplier = max(2, _env_int("RAG_CANDIDATE_MULTIPLIER", 8))
        candidate_count = max(top_k * candidate_multiplier, top_k)
        ranked = sorted(blended, key=lambda x: x[1], reverse=True)[:candidate_count]

        chunk_map = {c["chunk_id"]: c for c in chunks}
        reranked = rerank_candidates(query, ranked, chunk_map, top_k=top_k)

        if _env_bool("RAG_SUPPRESS_TOC", True):
            non_toc = [item for item in reranked if not chunk_map[item[0]].get("toc_like", False)]
            if len(non_toc) >= top_k:
                reranked = non_toc[:top_k]
            else:
                existing = {cid for cid, _ in non_toc}
                fill = [item for item in reranked if item[0] not in existing]
                reranked = (non_toc + fill)[:top_k]

        return [
            SearchResult(
                chunk_id=cid,
                score=score,
                text=chunk_map[cid]["text"],
                source_file=chunk_map[cid]["source_file"],
                page_start=chunk_map[cid]["page_start"],
                page_end=chunk_map[cid]["page_end"],
                section=chunk_map[cid].get("section"),
                toc_like=bool(chunk_map[cid].get("toc_like", False)),
            )
            for cid, score in reranked
        ]

    def answer(
        self,
        query: str,
        top_k: int = 6,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
    ) -> dict[str, Any]:
        results = self.search(query, top_k=top_k, bm25_weight=bm25_weight, vector_weight=vector_weight)
        context = []
        for i, r in enumerate(results, start=1):
            cite = f"[{r.source_file} p.{r.page_start}-{r.page_end}]"
            context.append(f"Context {i} {cite}\n{r.text}")

        answer_text = None
        llm_provider = resolve_llm_provider()
        if llm_provider != "none":
            answer_text = generate_answer(query, "\n\n".join(context), provider=llm_provider)

        if not answer_text:
            answer_text = _extractive_answer(query, results)

        return {
            "query": query,
            "answer": answer_text,
            "results": [r.__dict__ for r in results],
        }

    def _bm25_scores(self, query_tokens: list[str]) -> dict[str, float]:
        index = self.load()
        bm25 = index["bm25"]
        chunks = index["chunks"]
        doc_freqs = bm25["doc_freqs"]
        doc_count = bm25["doc_count"]
        avg_doc_len = bm25["avg_doc_len"] or 1.0
        k1 = bm25.get("k1", 1.5)
        b = bm25.get("b", 0.75)

        scores: dict[str, float] = {}
        for chunk in chunks:
            cid = chunk["chunk_id"]
            tf = chunk["term_freq"]
            doc_len = chunk["doc_len"]
            score = 0.0
            for term in query_tokens:
                if term not in tf:
                    continue
                df = doc_freqs.get(term, 0)
                idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
                term_freq = tf[term]
                denom = term_freq + k1 * (1 - b + b * doc_len / avg_doc_len)
                score += idf * ((term_freq * (k1 + 1)) / (denom + 1e-12))
            scores[cid] = score
        return scores

    def _build_chunks_from_pdfs(self, pdf_paths: list[Path], max_tokens: int, overlap_tokens: int) -> list[Chunk]:
        chunks: list[Chunk] = []
        for pdf_path in pdf_paths:
            pages = extract_pdf_pages(pdf_path)
            section = None
            page_toc_flags = {i + 1: is_toc_page(p) for i, p in enumerate(pages)}
            current_words: list[str] = []
            chunk_start_page = 1
            chunk_end_page = 1

            def flush_chunk() -> None:
                nonlocal current_words, chunk_start_page, chunk_end_page
                if not current_words:
                    return
                text = " ".join(current_words).strip()
                if not text:
                    return

                token_count = len(current_words)
                toc_votes = sum(1 for p in range(chunk_start_page, chunk_end_page + 1) if page_toc_flags.get(p, False))
                page_span = max(1, chunk_end_page - chunk_start_page + 1)
                toc_like = bool(toc_votes / page_span >= 0.5 or is_toc_section(section, text))

                chunk_id = f"{pdf_path.stem}-{chunk_start_page:04d}-{chunk_end_page:04d}-{len(chunks):05d}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=text,
                        source_file=pdf_path.name,
                        page_start=chunk_start_page,
                        page_end=chunk_end_page,
                        section=section,
                        token_count=token_count,
                        toc_like=toc_like,
                    )
                )

                overlap = current_words[-overlap_tokens:] if overlap_tokens > 0 else []
                current_words = overlap[:]
                chunk_start_page = chunk_end_page

            for page_num, page_text in enumerate(pages, start=1):
                normalized = normalize_text(page_text)
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
                for para in paragraphs:
                    heading = detect_heading(para)
                    if heading and heading != section and len(current_words) >= max(120, max_tokens // 3):
                        chunk_end_page = page_num
                        flush_chunk()
                    if heading:
                        section = heading

                    para_words = para.split()
                    if not para_words:
                        continue

                    if not current_words:
                        chunk_start_page = page_num

                    if len(current_words) + len(para_words) > max_tokens and current_words:
                        chunk_end_page = page_num
                        flush_chunk()
                        if not current_words:
                            chunk_start_page = page_num

                    current_words.extend(para_words)
                    chunk_end_page = page_num

                if len(current_words) >= int(max_tokens * 0.9):
                    flush_chunk()

            flush_chunk()

        return chunks


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    proc = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pages = [p for p in proc.stdout.split("\f") if p.strip()]
    if not pages:
        raise RuntimeError(f"No text extracted from {pdf_path}")
    return pages


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def normalize_text(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_heading(paragraph: str) -> str | None:
    first_line = paragraph.splitlines()[0].strip()
    if len(first_line) > 100:
        return None
    if HEADING_RE.match(first_line):
        return first_line
    return None


def is_toc_page(text: str) -> bool:
    lowered = text.lower()
    return "table of contents" in lowered


def is_toc_section(section: str | None, text: str) -> bool:
    if section and "table of contents" in section.lower():
        return True
    first_window = text[:1000].lower()
    return "table of contents" in first_window


def resolve_retrieval_weights(bm25_weight: float | None, vector_weight: float | None) -> tuple[float, float]:
    if bm25_weight is None:
        bm25_weight = _env_float("RAG_BM25_WEIGHT", 0.85)
    if vector_weight is None:
        vector_weight = _env_float("RAG_VECTOR_WEIGHT", 0.15)

    total = max(1e-9, bm25_weight + vector_weight)
    bm25_weight = bm25_weight / total
    vector_weight = vector_weight / total
    return bm25_weight, vector_weight


def rerank_candidates(
    query: str,
    ranked_candidates: list[tuple[str, float]],
    chunk_map: dict[str, dict[str, Any]],
    top_k: int,
) -> list[tuple[str, float]]:
    strategy = (os.getenv("RAG_RERANKER", "auto") or "auto").strip().lower()
    rerank_n = max(top_k, _env_int("RAG_RERANK_CANDIDATES", 30))
    candidates = ranked_candidates[:rerank_n]

    if strategy == "none":
        return candidates[:top_k]

    if strategy in {"auto", "cross-encoder"}:
        model = os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        ce = _load_cross_encoder(model)
        if ce is not None:
            reranked = _rerank_with_cross_encoder(ce, query, candidates, chunk_map)
            return reranked[:top_k]
        if strategy == "cross-encoder":
            print("[rerank] cross-encoder requested but unavailable; falling back to heuristic")

    reranked = sorted(
        candidates,
        key=lambda x: _rerank_heuristic(query, chunk_map[x[0]]["text"], x[1]),
        reverse=True,
    )
    return reranked[:top_k]


def _load_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return None

    cache = getattr(_load_cross_encoder, "_cache", {})
    if model_name in cache:
        return cache[model_name]

    try:
        model = CrossEncoder(model_name)
    except Exception as exc:
        print(f"[rerank] could not load cross-encoder model '{model_name}': {exc}")
        return None

    cache[model_name] = model
    setattr(_load_cross_encoder, "_cache", cache)
    return model


def _rerank_with_cross_encoder(ce, query: str, candidates: list[tuple[str, float]], chunk_map: dict[str, dict[str, Any]]) -> list[tuple[str, float]]:
    pairs = [(query, chunk_map[cid]["text"][:2200]) for cid, _ in candidates]
    try:
        ce_scores = ce.predict(pairs)
    except Exception as exc:
        print(f"[rerank] cross-encoder scoring failed: {exc}")
        return sorted(
            candidates,
            key=lambda x: _rerank_heuristic(query, chunk_map[x[0]]["text"], x[1]),
            reverse=True,
        )

    ce_list = [float(x) for x in ce_scores]
    ce_min = min(ce_list) if ce_list else 0.0
    ce_max = max(ce_list) if ce_list else 1.0
    denom = (ce_max - ce_min) or 1.0

    fused: list[tuple[str, float]] = []
    for (cid, base_score), ce_score in zip(candidates, ce_list):
        ce_norm = (ce_score - ce_min) / denom
        final = 0.75 * ce_norm + 0.25 * base_score
        if chunk_map[cid].get("toc_like", False):
            final -= _env_float("RAG_TOC_PENALTY", 0.35)
        fused.append((cid, final))

    return sorted(fused, key=lambda x: x[1], reverse=True)


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _token_overlap_bonus(query: str, text: str) -> float:
    q = set(tokenize(query))
    d = set(tokenize(text[:2500]))
    if not q:
        return 0.0
    return len(q & d) / len(q)


def _rerank_heuristic(query: str, text: str, base_score: float) -> float:
    query_terms = tokenize(query)
    if not query_terms:
        return base_score
    lowered = text.lower()
    phrase_bonus = 0.2 if " ".join(query_terms[:4]) in lowered else 0.0
    coverage = sum(1 for t in set(query_terms) if t in lowered) / len(set(query_terms))
    return base_score + 0.4 * coverage + phrase_bonus


def _extractive_answer(query: str, results: list[SearchResult]) -> str:
    if not results:
        return "I could not find relevant content in the indexed documents."
    lines = [
        "I do not have an LLM configured, so here are the most relevant excerpts:",
    ]
    for r in results[:3]:
        cite = f"[{r.source_file} p.{r.page_start}-{r.page_end}]"
        snippet = r.text[:420].strip().replace("\n", " ")
        lines.append(f"- {cite} {snippet}...")
    return "\n".join(lines)


def resolve_embedding_provider() -> str:
    explicit = (os.getenv("EMBEDDING_PROVIDER") or "").strip().lower()
    if explicit in {"openai", "bedrock", "ollama", "none"}:
        return explicit

    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("OLLAMA_EMBED_MODEL"):
        return "ollama"
    if os.getenv("BEDROCK_EMBED_MODEL") and (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")):
        return "bedrock"
    return "none"


def resolve_llm_provider() -> str:
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit in {"openai", "bedrock", "ollama", "none"}:
        return explicit

    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_MODEL"):
        return "ollama"
    if os.getenv("BEDROCK_CHAT_MODEL") and (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")):
        return "bedrock"
    return "none"


def default_embedding_model(provider: str) -> str | None:
    if provider == "openai":
        return os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    if provider == "ollama":
        return os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    if provider == "bedrock":
        return os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
    return None


def generate_answer(query: str, context: str, provider: str) -> str:
    prompt = (
        "Answer only from the provided context. If unavailable, say so. "
        "Cite sources inline in this format: [file p.start-end].\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}"
    )

    try:
        if provider == "openai":
            return _generate_answer_openai(prompt)
        if provider == "ollama":
            return _generate_answer_ollama(prompt)
        if provider == "bedrock":
            return _generate_answer_bedrock(prompt)
    except Exception as exc:  # pragma: no cover - defensive runtime fallback
        print(f"[answer] provider={provider} failed: {exc}")
    return ""


def embed_texts(texts: list[str], provider: str, model: str | None, batch_size: int = 32) -> list[list[float]]:
    if not texts:
        return []

    if provider == "openai":
        if not model:
            model = default_embedding_model("openai")
        return _embed_texts_openai(texts, model=model, batch_size=batch_size)
    if provider == "ollama":
        if not model:
            model = default_embedding_model("ollama")
        return _embed_texts_ollama(texts, model=model)
    if provider == "bedrock":
        if not model:
            model = default_embedding_model("bedrock")
        return _embed_texts_bedrock(texts, model=model)
    raise RuntimeError(f"Unknown embedding provider: {provider}")


def _embed_texts_openai(texts: list[str], model: str, batch_size: int = 32) -> list[list[float]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = json.dumps({"model": model, "input": batch}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        data = body.get("data", [])
        data = sorted(data, key=lambda item: item["index"])
        all_vectors.extend([row["embedding"] for row in data])

    return all_vectors


def _embed_texts_ollama(texts: list[str], model: str) -> list[list[float]]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama embeddings request failed: {exc.code} {detail}") from exc

    vectors = body.get("embeddings")
    if not isinstance(vectors, list) or not vectors:
        raise RuntimeError("Ollama did not return embeddings")
    return vectors


def _embed_texts_bedrock(texts: list[str], model: str) -> list[list[float]]:
    try:
        import boto3
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("boto3 is required for Bedrock support") from exc

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError("AWS_REGION or AWS_DEFAULT_REGION must be set for Bedrock")

    client = boto3.client("bedrock-runtime", region_name=region)
    vectors: list[list[float]] = []
    for text in texts:
        body = json.dumps({"inputText": text})
        response = client.invoke_model(
            modelId=model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        emb = payload.get("embedding")
        if not emb:
            raise RuntimeError("Bedrock embedding response missing `embedding`")
        vectors.append(emb)

    return vectors


def _generate_answer_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    payload = json.dumps(
        {
            "model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            "input": prompt,
            "max_output_tokens": 700,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    if "output_text" in body and body["output_text"]:
        return body["output_text"]

    output = body.get("output", [])
    texts: list[str] = []
    for item in output:
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    return "\n".join(t for t in texts if t).strip()


def _generate_answer_ollama(prompt: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_CHAT_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1"))

    payload = json.dumps(
        {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful budget analyst. Answer only from provided context and include citations.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "options": {"temperature": 0.1},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama chat request failed: {exc.code} {detail}") from exc

    return body.get("message", {}).get("content", "").strip()


def _generate_answer_bedrock(prompt: str) -> str:
    try:
        import boto3
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("boto3 is required for Bedrock support") from exc

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError("AWS_REGION or AWS_DEFAULT_REGION must be set for Bedrock")

    model = os.getenv("BEDROCK_CHAT_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    client = boto3.client("bedrock-runtime", region_name=region)

    response = client.converse(
        modelId=model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"temperature": 0.1, "maxTokens": 700},
    )
    content = response.get("output", {}).get("message", {}).get("content", [])
    texts = [c.get("text", "") for c in content if isinstance(c, dict)]
    return "\n".join(t for t in texts if t).strip()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
