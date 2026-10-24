#!/usr/bin/env python3
"""Validate Pipis JSONL datasets without requiring ML libraries."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "pipis_train.jsonl"
EVAL = ROOT / "data" / "pipis_eval.jsonl"


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError(f"{path.name}:{line_number}: expected an object with a string 'text' field")
        if not row["text"].strip():
            raise ValueError(f"{path.name}:{line_number}: text must not be empty")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: dataset is empty")
    return rows


def main() -> None:
    train = load_jsonl(TRAIN)
    evaluation = load_jsonl(EVAL)
    train_texts = {row["text"].strip() for row in train}
    eval_texts = {row["text"].strip() for row in evaluation}
    overlap = train_texts & eval_texts

    print(f"Train examples: {len(train)}")
    print(f"Eval examples:  {len(evaluation)}")
    print(f"Duplicate train texts: {len(train) - len(train_texts)}")
    print(f"Duplicate eval texts:  {len(evaluation) - len(eval_texts)}")
    if overlap:
        raise SystemExit(f"ERROR: {len(overlap)} exact text(s) appear in both train and eval.")
    print("JSONL format valid; no exact train/eval text overlap.")
    print("Reminder: this checks format and exact duplicates, not the quality of the examples.")


if __name__ == "__main__":
    main()
