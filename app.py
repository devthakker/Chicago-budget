#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import threading
import time
import csv
import io
import re
import json
from collections import defaultdict, deque
from pathlib import Path
import sys

from fastapi import FastAPI, Form, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chicago_budget_rag.engine import RAGEngine

app = FastAPI(title="Chicago Budget RAG")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
engine = RAGEngine(ROOT / "data/index")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
PDF_FILES = {p.name: p for p in ROOT.glob("*.pdf")}
SAMPLE_QUERIES = [
    "What grants mention ARPA?",
    "What is budgeted for the Office of the Mayor?",
    "Which funds are listed under grant funds?",
    "What does the ordinance say about new grants not included in the appropriation?",
    "Show examples of State Grant Fund entries.",
]

_RATE_LIMIT_ENABLED = (os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"})
_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "20"))
_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
_RATE_LIMIT_PATH = os.getenv("RATE_LIMIT_PATH", "/")
_RATE_LIMIT_METHOD = os.getenv("RATE_LIMIT_METHOD", "POST").upper()
_RATE_LIMIT_TRUST_PROXY = (os.getenv("RATE_LIMIT_TRUST_PROXY", "true").strip().lower() in {"1", "true", "yes", "on"})
_SITE_ENABLED = (os.getenv("SITE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"})
_SITE_DISABLED_REPO_URL = os.getenv("SITE_DISABLED_REPO_URL", "https://github.com")

_rate_limit_store: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    if _RATE_LIMIT_TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for", "").strip()
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip", "").strip()
        if real_ip:
            return real_ip
    return request.client.host if request.client else "unknown"


def _check_rate_limit(key: str) -> tuple[bool, int]:
    now = time.monotonic()
    window_start = now - _RATE_LIMIT_WINDOW_SECONDS

    with _rate_limit_lock:
        hits = _rate_limit_store[key]
        while hits and hits[0] <= window_start:
            hits.popleft()

        if len(hits) >= _RATE_LIMIT_MAX_REQUESTS:
            retry_after = max(1, int(math.ceil(_RATE_LIMIT_WINDOW_SECONDS - (now - hits[0]))))
            return False, retry_after

        hits.append(now)
        return True, 0


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if (
        _RATE_LIMIT_ENABLED
        and request.method.upper() == _RATE_LIMIT_METHOD
        and request.url.path == _RATE_LIMIT_PATH
    ):
        ip = _client_ip(request)
        allowed, retry_after = _check_rate_limit(f"{request.method}:{request.url.path}:{ip}")
        if not allowed:
            if request.headers.get("accept", "").lower().find("application/json") >= 0:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after_seconds": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )
            return HTMLResponse(
                status_code=429,
                content=templates.get_template("rate_limited.html").render(
                    {
                        "retry_after": retry_after,
                        "limit": _RATE_LIMIT_MAX_REQUESTS,
                        "window_seconds": _RATE_LIMIT_WINDOW_SECONDS,
                    }
                ),
                headers={"Retry-After": str(retry_after)},
            )
    return await call_next(request)


@app.middleware("http")
async def site_enabled_middleware(request: Request, call_next):
    if _SITE_ENABLED:
        return await call_next(request)

    path = request.url.path
    if path == "/health":
        return await call_next(request)

    return HTMLResponse(
        status_code=503,
        content=templates.get_template("site_disabled.html").render(
            {
                "repo_url": _SITE_DISABLED_REPO_URL,
            }
        ),
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "query": "",
            "answer": None,
            "results": [],
            "error": None,
            "sample_queries": SAMPLE_QUERIES,
        },
    )


@app.post("/", response_class=HTMLResponse)
async def ask(request: Request, query: str = Form(...)) -> HTMLResponse:
    query = query.strip()
    if not query:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "query": query,
                "answer": None,
                "results": [],
                "error": "Enter a question to search the budget documents.",
                "sample_queries": SAMPLE_QUERIES,
            },
        )

    try:
        payload = engine.answer(query, top_k=6)
        answer = payload["answer"]
        results = payload["results"]
        error = None
    except FileNotFoundError:
        answer = None
        results = []
        error = "Index not found. Run `python3 build_index.py` first."

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "query": query,
            "answer": answer,
            "results": results,
            "error": error,
            "sample_queries": SAMPLE_QUERIES,
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/pdf/{filename}")
async def get_pdf(filename: str):
    if filename not in PDF_FILES:
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(PDF_FILES[filename], media_type="application/pdf")


def _slug(value: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return (s[:max_len].strip("-") or "query")


def _to_markdown(query: str, payload: dict) -> str:
    lines = [
        "# Chicago Budget Query Export",
        "",
        f"## Query",
        query,
        "",
        "## Answer",
        payload.get("answer", ""),
        "",
        "## Sources",
    ]
    for i, r in enumerate(payload.get("results", []), start=1):
        lines.append(
            f"{i}. {r.get('source_file')} p.{r.get('page_start')}-{r.get('page_end')} "
            f"(score={r.get('score', 0):.3f})"
        )
        snippet = str(r.get("text", "")).replace("\n", " ").strip()
        lines.append(f"   - {snippet[:800]}{'...' if len(snippet) > 800 else ''}")
    return "\n".join(lines) + "\n"


def _to_csv(payload: dict) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["source_file", "page_start", "page_end", "score", "section", "snippet"])
    for r in payload.get("results", []):
        snippet = str(r.get("text", "")).replace("\n", " ").strip()
        writer.writerow(
            [
                r.get("source_file"),
                r.get("page_start"),
                r.get("page_end"),
                f"{float(r.get('score', 0)):.6f}",
                r.get("section") or "",
                snippet[:1000],
            ]
        )
    return out.getvalue()


@app.get("/export")
async def export_query(
    query: str = Query(..., min_length=1),
    fmt: str = Query("markdown"),
):
    fmt = fmt.strip().lower()
    if fmt not in {"markdown", "json", "csv"}:
        raise HTTPException(status_code=400, detail="fmt must be one of: markdown, json, csv")

    payload = engine.answer(query.strip(), top_k=6)
    base = _slug(query)

    if fmt == "json":
        body = json.dumps(payload, indent=2)
        return Response(
            content=body,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{base}.json"'},
        )
    if fmt == "csv":
        body = _to_csv(payload)
        return Response(
            content=body,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{base}.csv"'},
        )

    body = _to_markdown(query, payload)
    return Response(
        content=body,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{base}.md"'},
    )
