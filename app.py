#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
import sys

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
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

_RATE_LIMIT_ENABLED = (os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"})
_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "20"))
_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
_RATE_LIMIT_PATH = os.getenv("RATE_LIMIT_PATH", "/")
_RATE_LIMIT_METHOD = os.getenv("RATE_LIMIT_METHOD", "POST").upper()
_RATE_LIMIT_TRUST_PROXY = (os.getenv("RATE_LIMIT_TRUST_PROXY", "true").strip().lower() in {"1", "true", "yes", "on"})

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
