from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def write_jsonl(path: str | Path, rows: Iterator[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: str | Path, skip: int = 0) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            line = line.strip()
            if line:
                yield json.loads(line)
