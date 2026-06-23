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
import html
from collections import defaultdict, deque
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote_plus

from fastapi import FastAPI, Form, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chicago_budget_rag.engine import RAGEngine
from chicago_budget_rag.structured_budget import StructuredBudgetDataset

app = FastAPI(title="Chicago Budget RAG")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
engine = RAGEngine(ROOT / "data/index")
budget_dataset = StructuredBudgetDataset(ROOT)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
PDF_FILES = {p.name: p for p in ROOT.glob("*.pdf")}
FAVICON_FILE = ROOT / "static" / "favicon.ico"
ICON_FILE = ROOT / "static" / "icon.png"
SITE_NAME = "Chicago Budget Search"
BASE_URL = os.getenv("BASE_URL", "https://chicago-budget.thecommonnews.com").rstrip("/")
DEFAULT_DESCRIPTION = (
    "Search the Chicago FY2026 budget and grant ordinances in plain English, "
    "with page-level citations and direct links to the source PDFs."
)
SAMPLE_QUERIES = [
    "What grants mention ARPA?",
    "What is budgeted for the Office of the Mayor?",
    "Which funds are listed under grant funds?",
    "What does the ordinance say about new grants not included in the appropriation?",
    "Show examples of State Grant Fund entries.",
]
POPULAR_QUERY_PAGES = [
    "What grants mention ARPA?",
    "What is budgeted for the Office of the Mayor?",
    "Which funds are listed under grant funds?",
    "What does the ordinance say about new grants not included in the appropriation?",
    "Show examples of State Grant Fund entries.",
    "What does the grants ordinance table of contents cover?",
    "Which departments appear in the grants ordinance?",
    "What is in the State Grant Fund?",
    "What is in the Federal Grant Fund?",
    "What is in the Local Public and Private Grant Fund?",
    "Where does the budget discuss disaster recovery funding?",
    "What does the budget say about Neighborhoods Opportunity Fund revenue?",
    "What grant funds are appropriated for DCASE?",
    "What does the budget say about the American Rescue Plan Act local fiscal recovery fund?",
    "Which pages discuss the Mayor's Office in the grants ordinance?",
    "What does the budget say about the Disaster Recovery Fund?",
    "Which pages discuss entitlement funding in the grants ordinance?",
    "What does the budget say about the Corporate Fund?",
    "Which city departments appear at the start of the annual appropriation ordinance?",
    "What does the budget say about program income funding?",
]
GUIDES = [
    {
        "slug": "how-to-read-the-chicago-budget",
        "title": "How to Read the Chicago FY2026 Budget",
        "summary": "A plain-language guide to the major fund names, department sections, and how to use this search tool to navigate the ordinance faster.",
        "query": "How do I read the Chicago FY2026 budget?",
        "sections": [
            {
                "heading": "Start With The Two Documents",
                "body": "The Annual Appropriation Ordinance gives the broader budget structure, while the Grant Details Ordinance breaks out many grant-funded programs. If you are looking for a department's base appropriations, start with the annual appropriation document. If you are looking for a federal, state, or private grant-funded program, start with the grants ordinance.",
            },
            {
                "heading": "Use Fund Names As Anchors",
                "body": "Terms like Corporate Fund, State Grant Fund, Federal Grant Fund, Local Public and Private Grant Fund, and ARPA-related fund labels are useful search anchors. Many budget questions become easier once you search by fund name plus department or program.",
            },
            {
                "heading": "Follow The Citations",
                "body": "This site is designed to surface direct evidence. Use the page citations and PDF viewer to verify the answer and inspect surrounding context before quoting or relying on a figure publicly.",
            },
        ],
    },
    {
        "slug": "arpa-grants-in-the-chicago-budget",
        "title": "Where ARPA Appears in the Chicago FY2026 Budget",
        "summary": "A guide to finding ARPA, SLFRF, and related local fiscal recovery references across the budget and grants ordinances.",
        "query": "What grants mention ARPA?",
        "sections": [
            {
                "heading": "Search By Both ARPA And Program Labels",
                "body": "ARPA references may appear as ARPA, American Rescue Plan Act, SLFRF, or Local Fiscal Recovery Fund. Search using multiple variants when tracing references across budget and grant sections.",
            },
            {
                "heading": "Expect Both Summary And Line-Item Mentions",
                "body": "Some pages summarize ARPA-related totals while others show specific program or departmental line items. The most useful workflow is to start with the broad ARPA query, then narrow into department-specific questions.",
            },
        ],
    },
    {
        "slug": "mayors-office-budget-chicago-fy2026",
        "title": "How to Find the Mayor's Office Budget in Chicago FY2026",
        "summary": "A quick explainer for locating Mayor's Office budget entries and understanding where to look for grant-backed versus core appropriations.",
        "query": "What is budgeted for the Office of the Mayor?",
        "sections": [
            {
                "heading": "Start In The Annual Appropriation Ordinance",
                "body": "The Mayor's Office appears near the beginning of the annual appropriation document, which makes it one of the easiest departments to navigate directly from the top of the PDF.",
            },
            {
                "heading": "Separate Core Appropriations From Grant Detail",
                "body": "If a question is about overall department funding, use the annual appropriation document first. If the question is about a specific externally funded initiative, search the grants ordinance separately.",
            },
        ],
    },
]
GUIDES_BY_SLUG = {guide["slug"]: guide for guide in GUIDES}

_RATE_LIMIT_ENABLED = (os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"})
_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "20"))
_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
_RATE_LIMIT_PATH = os.getenv("RATE_LIMIT_PATH", "/")
_RATE_LIMIT_METHOD = os.getenv("RATE_LIMIT_METHOD", "POST").upper()
_RATE_LIMIT_TRUST_PROXY = (os.getenv("RATE_LIMIT_TRUST_PROXY", "true").strip().lower() in {"1", "true", "yes", "on"})
_SITE_ENABLED = (os.getenv("SITE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"})
_SITE_DISABLED_REPO_URL = os.getenv("SITE_DISABLED_REPO_URL", "https://github.com")
_SIMPLE_ANALYTICS_ENABLED = (os.getenv("SIMPLE_ANALYTICS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"})

_rate_limit_store: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = threading.Lock()


def _currency(value: int | float | None) -> str:
    value = 0 if value is None else value
    return f"${value:,.0f}"


def _compact_currency(value: int | float | None) -> str:
    value = 0 if value is None else float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000:
        return f"{sign}${value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{sign}${value / 1_000:.0f}K"
    return f"{sign}${value:,.0f}"


templates.env.filters["currency"] = _currency
templates.env.filters["compact_currency"] = _compact_currency


def _client_ip(request: Request) -> str:
    if _RATE_LIMIT_TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for", "").strip()
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip", "").strip()
        if real_ip:
            return real_ip
    return request.client.host if request.client else "unknown"


def _absolute_url(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{BASE_URL}{path}"


def _search_url(query: str) -> str:
    return f"/search?q={quote_plus(query)}"


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def _is_curated_query(query: str) -> bool:
    curated = {_normalize_query(q) for q in POPULAR_QUERY_PAGES}
    return _normalize_query(query) in curated


def _sample_query_links() -> list[dict[str, str]]:
    return [{"query": q, "href": _search_url(q)} for q in SAMPLE_QUERIES]


def _popular_query_links() -> list[dict[str, str]]:
    return [{"query": q, "href": _search_url(q)} for q in POPULAR_QUERY_PAGES]


def _featured_query_links(limit: int = 6, exclude: str | None = None) -> list[dict[str, str]]:
    normalized_exclude = _normalize_query(exclude or "")
    items: list[dict[str, str]] = []
    for q in POPULAR_QUERY_PAGES:
        if normalized_exclude and _normalize_query(q) == normalized_exclude:
            continue
        items.append({"query": q, "href": _search_url(q)})
        if len(items) >= limit:
            break
    return items


def _guide_links() -> list[dict[str, str]]:
    return [
        {
            "title": guide["title"],
            "summary": guide["summary"],
            "href": f"/guides/{guide['slug']}",
            "query_href": _search_url(guide["query"]),
        }
        for guide in GUIDES
    ]


def _load_budget_data() -> dict[str, Any]:
    return budget_dataset.load()


def _department_url(slug: str) -> str:
    return f"/departments/{slug}"


def _fund_url(slug: str) -> str:
    return f"/funds/{slug}"


def _record_pdf_url(record: dict[str, Any]) -> str:
    return f"/pdf/{record['source_file']}#page={record['page']}"


def _top_budget_links(limit: int = 8) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    data = _load_budget_data()
    departments = [
        {
            "name": item["name"],
            "code": item["code"],
            "href": _department_url(item["slug"]),
            "amount": item["appropriation_total"],
        }
        for item in data["departments"][:limit]
    ]
    funds = [
        {
            "name": item["name"],
            "code": item["code"],
            "href": _fund_url(item["slug"]),
            "amount": item["appropriation_total"],
        }
        for item in data["funds"][:limit]
    ]
    return departments, funds


def _tool_links() -> list[dict[str, str]]:
    return [
        {
            "title": "Budget Explorer",
            "href": "/explorer",
            "description": "Browse appropriations by department, fund, document, and keyword with direct PDF page links.",
        },
        {
            "title": "Budget Simulator",
            "href": "/simulator",
            "description": "Adjust department budgets, compare the net delta, and share a scenario URL.",
        },
    ]


def _sort_records(records: list[dict[str, Any]], sort: str) -> list[dict[str, Any]]:
    if sort == "department":
        return sorted(records, key=lambda r: (r["department_name"], -r["appropriation_total"], r["fund_name"]))
    if sort == "fund":
        return sorted(records, key=lambda r: (r["fund_name"], -r["appropriation_total"], r["department_name"]))
    if sort == "page":
        return sorted(records, key=lambda r: (r["source_file"], r["page"]))
    return sorted(records, key=lambda r: r["appropriation_total"], reverse=True)


def _filter_records(
    *,
    query: str = "",
    department: str = "",
    fund: str = "",
    document_type: str = "",
    sort: str = "appropriation",
) -> list[dict[str, Any]]:
    data = _load_budget_data()
    records = data["records"]
    query_norm = query.strip().lower()

    filtered = []
    for record in records:
        if department and record["department_slug"] != department:
            continue
        if fund and record["fund_slug"] != fund:
            continue
        if document_type and record["document_type"] != document_type:
            continue
        if query_norm:
            haystack = f"{record['searchable_text']} {' '.join(item['label'] for item in record['line_items'])}".lower()
            if query_norm not in haystack:
                continue
        filtered.append(record)

    return _sort_records(filtered, sort)


def _aggregate_for_records(records: list[dict[str, Any]], slug_key: str, code_key: str, name_key: str, href_fn) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        slug = record[slug_key]
        entry = grouped.setdefault(
            slug,
            {
                "slug": slug,
                "code": record[code_key],
                "name": record[name_key],
                "appropriation_total": 0,
                "record_count": 0,
                "href": href_fn(slug),
            },
        )
        entry["appropriation_total"] += int(record["appropriation_total"])
        entry["record_count"] += 1
    return sorted(grouped.values(), key=lambda item: item["appropriation_total"], reverse=True)


def _aggregate_line_items(records: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        for item in record.get("line_items", []):
            key = (item.get("line_code") or "", item.get("label") or "")
            entry = grouped.setdefault(
                key,
                {
                    "section_name": item.get("section_name"),
                    "line_code": item.get("line_code"),
                    "label": item.get("label"),
                    "amount": 0,
                    "count": 0,
                },
            )
            entry["amount"] += int(item.get("amount") or 0)
            entry["count"] += 1
    return sorted(grouped.values(), key=lambda item: item["amount"], reverse=True)[:limit]


def _simulator_departments(limit: int = 10) -> list[dict[str, Any]]:
    data = _load_budget_data()
    departments = []
    for item in data["departments"][:limit]:
        departments.append(
            {
                "slug": item["slug"],
                "code": item["code"],
                "name": item["name"],
                "amount": item["appropriation_total"],
                "href": _department_url(item["slug"]),
            }
        )
    return departments


def _json_ld_for_explorer(query: str, department: str, fund: str) -> str:
    title_bits = ["Chicago Budget Explorer"]
    if department:
        title_bits.append(department)
    if fund:
        title_bits.append(fund)
    if query:
        title_bits.append(query)
    payload = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": "Chicago FY2026 Budget Explorer",
        "description": "Structured appropriations extracted from the Chicago FY2026 annual appropriation and grant ordinances.",
        "url": _absolute_url("/explorer"),
        "keywords": ["Chicago budget", "appropriations", "grants", "departments", "funds"],
    }
    return json.dumps(payload)


def _json_ld_for_budget_entity(entity_type: str, entity: dict[str, Any], path: str) -> str:
    payload = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": f"{entity['name']} | {SITE_NAME}",
        "description": f"Structured Chicago FY2026 budget entries for {entity['name']}.",
        "url": _absolute_url(path),
        "keywords": [entity["name"], entity["code"], "Chicago budget"],
    }
    return json.dumps(payload)


def _json_ld_for_simulator() -> str:
    payload = {
        "@context": "https://schema.org",
        "@type": "WebApplication",
        "name": "Chicago Budget Simulator",
        "applicationCategory": "FinanceApplication",
        "url": _absolute_url("/simulator"),
        "description": "Adjust department-level Chicago budget scenarios and compare the net change.",
    }
    return json.dumps(payload)


def _json_ld_for_home() -> str:
    payload = {
        "@context": "https://schema.org",
        "@type": "WebSite",
        "name": SITE_NAME,
        "url": BASE_URL,
        "description": DEFAULT_DESCRIPTION,
        "potentialAction": {
            "@type": "SearchAction",
            "target": f"{BASE_URL}/search?q={{search_term_string}}",
            "query-input": "required name=search_term_string",
        },
    }
    return json.dumps(payload)


def _json_ld_for_search(query: str, results: list[dict]) -> str:
    payload = {
        "@context": "https://schema.org",
        "@type": "SearchResultsPage",
        "name": f"{query} | {SITE_NAME}",
        "url": _absolute_url(_search_url(query)),
        "mainEntity": [
            {
                "@type": "CreativeWork",
                "name": f"{row['source_file']} pages {row['page_start']}-{row['page_end']}",
                "url": _absolute_url(f"/pdf/{row['source_file']}#page={row['page_start']}"),
            }
            for row in results[:5]
        ],
    }
    return json.dumps(payload)


def _json_ld_for_guide(guide: dict) -> str:
    payload = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": guide["title"],
        "description": guide["summary"],
        "url": _absolute_url(f"/guides/{guide['slug']}"),
        "author": {"@type": "Organization", "name": SITE_NAME},
    }
    return json.dumps(payload)


def _format_inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    return escaped


def _render_answer_markdown(text: str | None) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    parts: list[str] = []
    paragraph: list[str] = []
    list_stack: list[tuple[str, int]] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            joined = " ".join(chunk.strip() for chunk in paragraph if chunk.strip())
            if joined:
                parts.append(f"<p>{_format_inline_markdown(joined)}</p>")
            paragraph = []

    def close_lists(min_indent: int = -1) -> None:
        while list_stack and list_stack[-1][1] >= min_indent:
            tag, _indent = list_stack.pop()
            parts.append(f"</li></{tag}>")

    def ensure_list(tag: str, indent: int) -> None:
        while list_stack and indent < list_stack[-1][1]:
            old_tag, _old_indent = list_stack.pop()
            parts.append(f"</li></{old_tag}>")
        if list_stack and indent == list_stack[-1][1] and tag != list_stack[-1][0]:
            old_tag, _old_indent = list_stack.pop()
            parts.append(f"</li></{old_tag}>")
        if not list_stack or indent > list_stack[-1][1] or tag != list_stack[-1][0]:
            parts.append(f"<{tag}><li>")
            list_stack.append((tag, indent))
        else:
            parts.append("</li><li>")

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            flush_paragraph()
            close_lists(0)
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match:
            flush_paragraph()
            close_lists(0)
            level = len(heading_match.group(1))
            parts.append(f"<h{level + 2}>{_format_inline_markdown(heading_match.group(2).strip())}</h{level + 2}>")
            continue

        ordered_match = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        unordered_match = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        if ordered_match or unordered_match:
            flush_paragraph()
            if ordered_match:
                indent = len(ordered_match.group(1))
                content = ordered_match.group(3)
                ensure_list("ol", indent)
            else:
                indent = len(unordered_match.group(1))
                content = unordered_match.group(2)
                ensure_list("ul", indent)
            parts.append(_format_inline_markdown(content.strip()))
            continue

        if list_stack:
            parts.append(f"<br>{_format_inline_markdown(line.strip())}")
            continue

        paragraph.append(line)

    flush_paragraph()
    close_lists(0)
    return "".join(parts)


def _render_home(request: Request, query: str = "", answer=None, results=None, error=None) -> HTMLResponse:
    top_departments, top_funds = _top_budget_links()
    response = templates.TemplateResponse(
        request,
        "index.html",
        {
            "query": query,
            "answer": answer,
            "answer_html": _render_answer_markdown(answer),
            "results": results or [],
            "error": error,
            "sample_queries": _sample_query_links(),
            "popular_queries": _popular_query_links(),
            "featured_queries": _featured_query_links(limit=6),
            "more_popular_queries": _popular_query_links()[6:],
            "guides": _guide_links(),
            "top_departments": top_departments,
            "top_funds": top_funds,
            "tool_links": _tool_links(),
            "base_url": BASE_URL,
            "page_title": SITE_NAME,
            "meta_description": DEFAULT_DESCRIPTION,
            "canonical_url": BASE_URL,
            "robots_value": "index,follow",
            "json_ld": _json_ld_for_home(),
            "page_kind": "home",
            "analytics_payload": json.dumps({"page_type": "home"}),
        },
    )
    return response


def _render_search(request: Request, query: str, answer, results, error) -> HTMLResponse:
    indexable = _is_curated_query(query)
    top_departments, top_funds = _top_budget_links(limit=6)
    title = f"{query} | {SITE_NAME}"
    description = (
        f"Search results for '{query}' across Chicago FY2026 budget documents, "
        "with page-level citations and direct PDF links."
    )
    response = templates.TemplateResponse(
        request,
        "index.html",
        {
            "query": query,
            "answer": answer,
            "answer_html": _render_answer_markdown(answer),
            "results": results or [],
            "error": error,
            "sample_queries": _sample_query_links(),
            "popular_queries": _popular_query_links(),
            "featured_queries": _featured_query_links(limit=5, exclude=query),
            "more_popular_queries": [],
            "guides": _guide_links(),
            "top_departments": top_departments,
            "top_funds": top_funds,
            "tool_links": _tool_links(),
            "base_url": BASE_URL,
            "page_title": title,
            "meta_description": description,
            "canonical_url": _absolute_url(_search_url(query)),
            "robots_value": "index,follow" if indexable else "noindex,follow",
            "json_ld": _json_ld_for_search(query, results or []),
            "page_kind": "search",
            "analytics_payload": json.dumps({"page_type": "search", "query": query, "indexable": indexable}),
        },
    )
    return response


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
    return _render_home(request)


@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = Query("", alias="q")) -> HTMLResponse:
    query = q.strip()
    if not query:
        return _render_home(request)

    try:
        payload = engine.answer(query, top_k=6)
        answer = payload["answer"]
        results = payload["results"]
        error = None
    except FileNotFoundError:
        answer = None
        results = []
        error = "Index not found. Run `python3 build_index.py` first."

    return _render_search(request, query, answer, results, error)


@app.post("/", response_class=HTMLResponse)
async def ask(query: str = Form(...)) -> HTMLResponse:
    query = query.strip()
    if not query:
        return RedirectResponse(url="/", status_code=303)
    return RedirectResponse(url=_search_url(query), status_code=303)


@app.get("/guides", response_class=HTMLResponse)
async def guides_index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "guides.html",
        {
            "guides": _guide_links(),
            "base_url": BASE_URL,
            "page_title": f"Chicago Budget Guides | {SITE_NAME}",
            "meta_description": "Plain-language guides for understanding the Chicago FY2026 budget and grant ordinances.",
            "canonical_url": _absolute_url("/guides"),
            "robots_value": "index,follow",
            "json_ld": json.dumps(
                {
                    "@context": "https://schema.org",
                    "@type": "CollectionPage",
                    "name": "Chicago Budget Guides",
                    "url": _absolute_url("/guides"),
                }
            ),
            "analytics_payload": json.dumps({"page_type": "guides_index"}),
        },
    )


@app.get("/guides/{slug}", response_class=HTMLResponse)
async def guide_detail(request: Request, slug: str) -> HTMLResponse:
    guide = GUIDES_BY_SLUG.get(slug)
    if not guide:
        raise HTTPException(status_code=404, detail="Guide not found")
    return templates.TemplateResponse(
        request,
        "guide.html",
        {
            "guide": guide,
            "guide_query_href": _search_url(guide["query"]),
            "related_queries": _popular_query_links()[:6],
            "base_url": BASE_URL,
            "page_title": f"{guide['title']} | {SITE_NAME}",
            "meta_description": guide["summary"],
            "canonical_url": _absolute_url(f"/guides/{slug}"),
            "robots_value": "index,follow",
            "json_ld": _json_ld_for_guide(guide),
            "analytics_payload": json.dumps({"page_type": "guide", "slug": slug}),
        },
    )


@app.get("/explorer", response_class=HTMLResponse)
async def explorer(
    request: Request,
    q: str = Query("", alias="q"),
    department: str = Query(""),
    fund: str = Query(""),
    document_type: str = Query(""),
    sort: str = Query("appropriation"),
) -> HTMLResponse:
    data = _load_budget_data()
    records = _filter_records(
        query=q,
        department=department,
        fund=fund,
        document_type=document_type,
        sort=sort,
    )
    department_obj = next((item for item in data["departments"] if item["slug"] == department), None) if department else None
    fund_obj = next((item for item in data["funds"] if item["slug"] == fund), None) if fund else None
    total = sum(int(item["appropriation_total"]) for item in records)
    indexable = not any([q.strip(), department, fund, document_type])
    records_view = []
    for record in records[:150]:
        record_copy = dict(record)
        record_copy["pdf_url"] = _record_pdf_url(record)
        record_copy["department_href"] = _department_url(record["department_slug"])
        record_copy["fund_href"] = _fund_url(record["fund_slug"])
        records_view.append(record_copy)

    top_departments = _aggregate_for_records(records, "department_slug", "department_code", "department_name", _department_url)[:8]
    top_funds = _aggregate_for_records(records, "fund_slug", "fund_code", "fund_name", _fund_url)[:8]
    return templates.TemplateResponse(
        request,
        "explorer.html",
        {
            "query": q.strip(),
            "department": department,
            "fund": fund,
            "document_type": document_type,
            "sort": sort,
            "records": records_view,
            "record_count": len(records),
            "total_amount": total,
            "department_options": data["departments"],
            "fund_options": data["funds"],
            "department_obj": department_obj,
            "fund_obj": fund_obj,
            "top_departments": top_departments,
            "top_funds": top_funds,
            "tool_links": _tool_links(),
            "base_url": BASE_URL,
            "page_title": "Chicago Budget Explorer | Chicago Budget Search",
            "meta_description": "Explore structured Chicago FY2026 budget appropriations by department, fund, program, and document.",
            "canonical_url": _absolute_url("/explorer"),
            "robots_value": "index,follow" if indexable else "noindex,follow",
            "json_ld": _json_ld_for_explorer(q.strip(), department_obj["name"] if department_obj else "", fund_obj["name"] if fund_obj else ""),
            "analytics_payload": json.dumps({"page_type": "explorer", "query": q.strip(), "department": department, "fund": fund}),
        },
    )


@app.get("/departments/{slug}", response_class=HTMLResponse)
async def department_detail(request: Request, slug: str) -> HTMLResponse:
    data = _load_budget_data()
    entity = next((item for item in data["departments"] if item["slug"] == slug), None)
    if entity is None:
        raise HTTPException(status_code=404, detail="Department not found")

    records = _filter_records(department=slug, sort="appropriation")
    fund_breakdown = _aggregate_for_records(records, "fund_slug", "fund_code", "fund_name", _fund_url)[:12]
    top_line_items = _aggregate_line_items(records, limit=15)
    records_view = []
    for record in records[:20]:
        record_copy = dict(record)
        record_copy["pdf_url"] = _record_pdf_url(record)
        record_copy["fund_href"] = _fund_url(record["fund_slug"])
        records_view.append(record_copy)

    return templates.TemplateResponse(
        request,
        "entity_detail.html",
        {
            "entity_type": "department",
            "entity": entity,
            "records": records_view,
            "records_total_count": len(records),
            "breakdown_title": "Funding by fund",
            "breakdown_items": fund_breakdown,
            "top_line_items": top_line_items,
            "search_href": _search_url(f"What is budgeted for {entity['name']}?"),
            "explorer_href": f"/explorer?department={slug}",
            "base_url": BASE_URL,
            "page_title": f"{entity['name']} Budget | {SITE_NAME}",
            "meta_description": f"Structured Chicago FY2026 appropriations for {entity['name']}, with line items and source page links.",
            "canonical_url": _absolute_url(_department_url(slug)),
            "robots_value": "index,follow",
            "json_ld": _json_ld_for_budget_entity("department", entity, _department_url(slug)),
            "analytics_payload": json.dumps({"page_type": "department", "slug": slug}),
        },
    )


@app.get("/funds/{slug}", response_class=HTMLResponse)
async def fund_detail(request: Request, slug: str) -> HTMLResponse:
    data = _load_budget_data()
    entity = next((item for item in data["funds"] if item["slug"] == slug), None)
    if entity is None:
        raise HTTPException(status_code=404, detail="Fund not found")

    records = _filter_records(fund=slug, sort="appropriation")
    department_breakdown = _aggregate_for_records(records, "department_slug", "department_code", "department_name", _department_url)[:12]
    top_line_items = _aggregate_line_items(records, limit=15)
    records_view = []
    for record in records[:20]:
        record_copy = dict(record)
        record_copy["pdf_url"] = _record_pdf_url(record)
        record_copy["department_href"] = _department_url(record["department_slug"])
        records_view.append(record_copy)

    return templates.TemplateResponse(
        request,
        "entity_detail.html",
        {
            "entity_type": "fund",
            "entity": entity,
            "records": records_view,
            "records_total_count": len(records),
            "breakdown_title": "Funding by department",
            "breakdown_items": department_breakdown,
            "top_line_items": top_line_items,
            "search_href": _search_url(f"What does the budget say about the {entity['name']}?"),
            "explorer_href": f"/explorer?fund={slug}",
            "base_url": BASE_URL,
            "page_title": f"{entity['name']} | {SITE_NAME}",
            "meta_description": f"Structured Chicago FY2026 budget entries for the {entity['name']}.",
            "canonical_url": _absolute_url(_fund_url(slug)),
            "robots_value": "index,follow",
            "json_ld": _json_ld_for_budget_entity("fund", entity, _fund_url(slug)),
            "analytics_payload": json.dumps({"page_type": "fund", "slug": slug}),
        },
    )


@app.get("/simulator", response_class=HTMLResponse)
async def simulator(request: Request) -> HTMLResponse:
    scenario_departments = _simulator_departments(limit=10)
    baseline_total = sum(int(item["amount"]) for item in scenario_departments)
    simulator_payload = {
        "departments": scenario_departments,
        "baseline_total": baseline_total,
    }
    faq_items = [
        {
            "question": "What does this simulator represent?",
            "answer": "It shows a simplified department-level scenario model built from structured FY2026 appropriation totals parsed from the annual appropriation and grants ordinances.",
        },
        {
            "question": "Does it update the official budget?",
            "answer": "No. It is an educational scenario tool that helps users compare changes and share a budget tradeoff idea.",
        },
    ]
    return templates.TemplateResponse(
        request,
        "simulator.html",
        {
            "simulator_payload": json.dumps(simulator_payload),
            "departments": scenario_departments,
            "baseline_total": baseline_total,
            "faq_items": faq_items,
            "base_url": BASE_URL,
            "page_title": f"Chicago Budget Simulator | {SITE_NAME}",
            "meta_description": "Simulate department-level changes to the Chicago FY2026 budget and share the resulting scenario.",
            "canonical_url": _absolute_url("/simulator"),
            "robots_value": "index,follow",
            "json_ld": _json_ld_for_simulator(),
            "analytics_payload": json.dumps({"page_type": "simulator"}),
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/favicon.ico")
async def favicon() -> FileResponse:
    if not FAVICON_FILE.exists():
        raise HTTPException(status_code=404, detail="favicon not found")
    return FileResponse(FAVICON_FILE, media_type="image/x-icon")


@app.get("/icon.png")
async def icon_png() -> FileResponse:
    if not ICON_FILE.exists():
        raise HTTPException(status_code=404, detail="icon not found")
    return FileResponse(ICON_FILE, media_type="image/png")


@app.get("/robots.txt")
async def robots_txt() -> Response:
    body = "\n".join(
        [
            "User-agent: *",
            "Allow: /",
            "Disallow: /export",
            "Disallow: /analytics/event",
            f"Sitemap: {BASE_URL}/sitemap.xml",
            "",
        ]
    )
    return Response(content=body, media_type="text/plain")


@app.get("/sitemap.xml")
async def sitemap_xml() -> Response:
    data = _load_budget_data()
    urls = [BASE_URL, _absolute_url("/guides"), _absolute_url("/explorer"), _absolute_url("/simulator")]
    urls.extend(_absolute_url(f"/guides/{guide['slug']}") for guide in GUIDES)
    urls.extend(_absolute_url(_search_url(query)) for query in POPULAR_QUERY_PAGES)
    urls.extend(_absolute_url(_department_url(item["slug"])) for item in data["departments"])
    urls.extend(_absolute_url(_fund_url(item["slug"])) for item in data["funds"])
    urls.extend(_absolute_url(f"/pdf/{filename}") for filename in PDF_FILES)
    xml_items = []
    for url in urls:
        xml_items.append(f"<url><loc>{url}</loc></url>")
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + "".join(xml_items)
        + "</urlset>"
    )
    return Response(content=body, media_type="application/xml")


@app.post("/analytics/event")
async def analytics_event(request: Request) -> Response:
    if not _SIMPLE_ANALYTICS_ENABLED:
        return Response(status_code=204)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    event = {
        "event": payload.get("event"),
        "page_type": payload.get("page_type"),
        "query": payload.get("query"),
        "slug": payload.get("slug"),
        "href": payload.get("href"),
        "referrer": request.headers.get("referer"),
        "user_agent": request.headers.get("user-agent"),
    }
    print("[analytics]", json.dumps(event, ensure_ascii=True))
    return Response(status_code=204)


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
