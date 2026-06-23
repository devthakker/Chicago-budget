from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine import extract_pdf_pages, normalize_text


DEPARTMENT_RE = re.compile(r"^(?P<code>\d{3})\s*-\s+(?P<name>.+)$")
HEADER_CODE_RE = re.compile(r"^(?P<code>[0-9A-Z]{4})\s*-\s+(?P<name>.+)$")
TUPLE_RE = re.compile(r"^\((?P<tuple>[0-9A-Z/]+)\)$")
AMOUNT_RE = re.compile(r"\$?\(?\d[\d,]*\)?$")
NUMERIC_RE = re.compile(r"\(?\$?([\d,]+)\)?$")
LINE_CODE_RE = re.compile(r"^(?P<code>[0-9A-Z]{4})\s+(?P<body>.+)$")


@dataclass
class ParsedRecord:
    record_id: str
    slug: str
    source_file: str
    document_type: str
    page: int
    department_code: str
    department_name: str
    department_slug: str
    fund_code: str
    fund_name: str
    fund_slug: str
    office_code: str
    office_name: str
    office_slug: str
    program_code: str | None
    program_name: str | None
    program_slug: str | None
    tuple_code: str | None
    appropriation_total: int
    fund_total: int | None
    department_total: int | None
    line_items: list[dict[str, Any]]
    section_totals: list[dict[str, Any]]
    section_codes: list[str]
    searchable_text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "slug": self.slug,
            "source_file": self.source_file,
            "document_type": self.document_type,
            "page": self.page,
            "department_code": self.department_code,
            "department_name": self.department_name,
            "department_slug": self.department_slug,
            "fund_code": self.fund_code,
            "fund_name": self.fund_name,
            "fund_slug": self.fund_slug,
            "office_code": self.office_code,
            "office_name": self.office_name,
            "office_slug": self.office_slug,
            "program_code": self.program_code,
            "program_name": self.program_name,
            "program_slug": self.program_slug,
            "tuple_code": self.tuple_code,
            "appropriation_total": self.appropriation_total,
            "fund_total": self.fund_total,
            "department_total": self.department_total,
            "line_items": self.line_items,
            "section_totals": self.section_totals,
            "section_codes": self.section_codes,
            "searchable_text": self.searchable_text,
        }


class StructuredBudgetDataset:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.data_dir = self.root / "data" / "structured"
        self.data_file = self.data_dir / "budget_dataset.json"
        self._cache: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        if self._cache is None:
            if not self.data_file.exists():
                self.build()
            self._cache = json.loads(self.data_file.read_text())
        return self._cache

    def build(self, pdf_paths: list[Path] | None = None) -> dict[str, Any]:
        pdf_paths = pdf_paths or sorted(self.root.glob("*.pdf"))
        records: list[dict[str, Any]] = []
        for pdf_path in pdf_paths:
            records.extend(_parse_pdf(pdf_path))

        departments = _aggregate_entities(records, key="department_slug", code_key="department_code", name_key="department_name")
        funds = _aggregate_entities(records, key="fund_slug", code_key="fund_code", name_key="fund_name")
        offices = _aggregate_entities(records, key="office_slug", code_key="office_code", name_key="office_name")
        programs = _aggregate_programs(records)

        dataset = {
            "stats": {
                "record_count": len(records),
                "department_count": len(departments),
                "fund_count": len(funds),
                "office_count": len(offices),
                "program_count": len(programs),
                "documents": [p.name for p in pdf_paths],
            },
            "records": records,
            "departments": departments,
            "funds": funds,
            "offices": offices,
            "programs": programs,
            "top_departments": sorted(departments, key=lambda item: item["appropriation_total"], reverse=True)[:12],
            "top_funds": sorted(funds, key=lambda item: item["appropriation_total"], reverse=True)[:12],
        }
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_file.write_text(json.dumps(dataset))
        self._cache = dataset
        return dataset


def _parse_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    pages = extract_pdf_pages(pdf_path)
    records: list[dict[str, Any]] = []
    document_type = "grants" if "grant" in pdf_path.name.lower() else "annual"

    for page_num, page_text in enumerate(pages, start=1):
        if "Appropriation Total" not in page_text:
            continue
        parsed = _parse_page(pdf_path.name, document_type, page_num, page_text)
        if parsed is not None:
            records.append(parsed.as_dict())

    return records


def _parse_page(source_file: str, document_type: str, page_num: int, page_text: str) -> ParsedRecord | None:
    normalized = normalize_text(page_text)
    raw_lines = [line.rstrip() for line in normalized.splitlines()]
    lines = [line.strip() for line in raw_lines if line.strip()]

    dept = None
    headers: list[tuple[str, str]] = []
    tuple_code = None
    for line in lines[:18]:
        if dept is None:
            m = DEPARTMENT_RE.match(line)
            if m:
                dept = (m.group("code"), _clean_header_name(m.group("name")))
                continue
        m = HEADER_CODE_RE.match(line)
        if m:
            headers.append((m.group("code"), _clean_header_name(m.group("name"))))
            continue
        m = TUPLE_RE.match(line)
        if m:
            tuple_code = m.group("tuple")
            break

    if dept is None or len(headers) < 2:
        return None

    fund_code, fund_name = headers[0]
    office_code, office_name = headers[1]
    program_code = headers[2][0] if len(headers) >= 3 else None
    program_name = headers[2][1] if len(headers) >= 3 else None

    appropriation_start = next((i for i, line in enumerate(lines) if line.lower().startswith("appropriations")), None)
    if appropriation_start is None:
        return None

    body_lines: list[str] = []
    for line in lines[appropriation_start + 1 :]:
        lowered = line.lower()
        if lowered.startswith("positions and salaries") or lowered.startswith("annual appropriation ordinance"):
            break
        body_lines.append(line)

    line_items, section_totals, appropriation_total, fund_total, department_total = _parse_appropriation_body(body_lines)
    if appropriation_total is None:
        return None

    department_code, department_name = dept
    department_slug = slugify(f"{department_code}-{department_name}")
    fund_slug = slugify(f"{fund_code}-{fund_name}")
    office_slug = slugify(f"{office_code}-{office_name}")
    program_slug = slugify(f"{program_code}-{program_name}") if program_code and program_name else None

    search_bits = [
        department_code,
        department_name,
        fund_code,
        fund_name,
        office_code,
        office_name,
        program_code or "",
        program_name or "",
        " ".join(item["label"] for item in line_items[:12]),
    ]
    slug_parts = [department_code, fund_code, office_code]
    if program_code:
        slug_parts.append(program_code)
    slug_parts.append(str(page_num))

    return ParsedRecord(
        record_id=f"{source_file}:{page_num}:{'-'.join(slug_parts)}",
        slug=slugify("-".join(slug_parts)),
        source_file=source_file,
        document_type=document_type,
        page=page_num,
        department_code=department_code,
        department_name=department_name,
        department_slug=department_slug,
        fund_code=fund_code,
        fund_name=fund_name,
        fund_slug=fund_slug,
        office_code=office_code,
        office_name=office_name,
        office_slug=office_slug,
        program_code=program_code,
        program_name=program_name,
        program_slug=program_slug,
        tuple_code=tuple_code,
        appropriation_total=appropriation_total,
        fund_total=fund_total,
        department_total=department_total,
        line_items=line_items,
        section_totals=section_totals,
        section_codes=sorted({item["section_code"] for item in line_items if item.get("section_code")}),
        searchable_text=" ".join(bit for bit in search_bits if bit),
    )


def _parse_appropriation_body(lines: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int | None, int | None, int | None]:
    line_items: list[dict[str, Any]] = []
    section_totals: list[dict[str, Any]] = []
    appropriation_total = None
    fund_total = None
    department_total = None
    current_section_code: str | None = None
    current_section_name: str | None = None
    pending: dict[str, Any] | None = None

    def finalize_pending() -> None:
        nonlocal pending
        if pending is None:
            return
        if pending.get("amount") is not None:
            line_items.append(pending)
        pending = None

    for line in lines:
        if not line:
            continue
        if line.startswith("Fund Total"):
            finalize_pending()
            fund_total = _amount_from_line(line)
            continue
        if line.startswith("Department Total"):
            finalize_pending()
            department_total = _amount_from_line(line)
            continue
        if line.startswith("Appropriation Total"):
            finalize_pending()
            appropriation_total = _amount_from_line(line)
            continue

        match = LINE_CODE_RE.match(line)
        if match:
            code = match.group("code")
            body = match.group("body").strip()
            amount = _amount_from_line(body)
            label = _strip_amount(body)
            label = label.rstrip("*").strip()

            if " - Total" in label:
                finalize_pending()
                section_totals.append(
                    {
                        "section_code": code,
                        "section_name": label.replace(" - Total", "").strip(),
                        "amount": amount or 0,
                    }
                )
                continue

            if amount is None and pending is None:
                finalize_pending()
                current_section_code = code
                current_section_name = label
                continue

            finalize_pending()
            pending = {
                "section_code": current_section_code,
                "section_name": current_section_name,
                "line_code": code,
                "label": label,
                "amount": amount,
            }
            continue

        if pending is not None:
            extra_amount = _amount_from_line(line)
            extra_label = _strip_amount(line).strip()
            if extra_label:
                pending["label"] = f"{pending['label']} {extra_label}".strip()
            if extra_amount is not None:
                pending["amount"] = extra_amount
                finalize_pending()

    finalize_pending()
    return line_items, section_totals, appropriation_total, fund_total, department_total


def _aggregate_entities(records: list[dict[str, Any]], key: str, code_key: str, name_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        slug = record[key]
        entry = grouped.setdefault(
            slug,
            {
                "slug": slug,
                "code": record[code_key],
                "name": record[name_key],
                "appropriation_total": 0,
                "documents": set(),
                "record_count": 0,
                "pages": [],
                "sample_programs": [],
            },
        )
        entry["appropriation_total"] += int(record["appropriation_total"])
        entry["documents"].add(record["document_type"])
        entry["record_count"] += 1
        entry["pages"].append({"source_file": record["source_file"], "page": record["page"]})
        if record.get("program_name") and len(entry["sample_programs"]) < 8:
            entry["sample_programs"].append(record["program_name"])

    final = []
    for entry in grouped.values():
        entry["documents"] = sorted(entry["documents"])
        entry["pages"] = entry["pages"][:12]
        final.append(entry)
    return sorted(final, key=lambda item: item["appropriation_total"], reverse=True)


def _aggregate_programs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        slug = record["program_slug"] or record["slug"]
        name = record["program_name"] or record["office_name"]
        entry = grouped.setdefault(
            slug,
            {
                "slug": slug,
                "name": name,
                "code": record["program_code"] or record["office_code"],
                "department_name": record["department_name"],
                "fund_name": record["fund_name"],
                "appropriation_total": 0,
                "record_count": 0,
            },
        )
        entry["appropriation_total"] += int(record["appropriation_total"])
        entry["record_count"] += 1
    return sorted(grouped.values(), key=lambda item: item["appropriation_total"], reverse=True)


def _amount_from_line(line: str) -> int | None:
    match = NUMERIC_RE.search(line.replace("$", "").strip())
    if not match:
        return None
    digits = match.group(1).replace(",", "")
    if not digits.isdigit():
        return None
    value = int(digits)
    if "(" in line and ")" in line:
        return -value
    return value


def _strip_amount(text: str) -> str:
    return re.sub(r"\s+\$?\(?\d[\d,]*\)?\s*$", "", text).strip()


def _clean_header_name(name: str) -> str:
    return re.sub(r"\s+-\s+Continued$", "", name, flags=re.IGNORECASE).strip()


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")
