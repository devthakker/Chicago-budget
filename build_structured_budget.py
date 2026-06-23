#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chicago_budget_rag.structured_budget import StructuredBudgetDataset


def main() -> None:
    dataset = StructuredBudgetDataset(ROOT).build()
    print(f"Structured dataset written to {ROOT / 'data' / 'structured' / 'budget_dataset.json'}")
    print(f"Records: {dataset['stats']['record_count']}")
    print(f"Departments: {dataset['stats']['department_count']}")
    print(f"Funds: {dataset['stats']['fund_count']}")


if __name__ == "__main__":
    main()
