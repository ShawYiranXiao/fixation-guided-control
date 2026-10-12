#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check the public source without importing model or interface modules."""

import ast
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", "__pycache__", ".ruff_cache", ".pytest_cache"}
PATTERNS = {
    "Chinese text": re.compile(r"[\u3400-\u9fff]"),
    "private machine path": re.compile(r"/(?:Users|home|Volumes)/"),
    "lab endpoint": re.compile(r"128[.]111[.][0-9]+[.][0-9]+"),
    "embedded Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{28,}"),
    "embedded API key": re.compile(r"sk-[A-Za-z0-9_-]{24,}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
FORBIDDEN_SUFFIXES = {".npy", ".npz", ".parquet", ".arrow", ".pt", ".pth", ".bin", ".safetensors"}


def check():
    errors = []
    count = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or IGNORED.intersection(path.relative_to(ROOT).parts):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix in FORBIDDEN_SUFFIXES or path.name in {".env", "labels.csv"}:
            errors.append(f"{relative}: generated or private artifact")
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            errors.append(f"{relative}: unexpected large source artifact")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            errors.append(f"{relative}: unexpected binary file")
            continue
        count += 1
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{relative}: {name}")
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=relative)
            except SyntaxError as error:
                errors.append(f"{relative}:{error.lineno}: {error.msg}")
            if "SPDX-License-Identifier:" not in text:
                errors.append(f"{relative}: missing license identifier")
    for required in [
        "README.md",
        "AUTHORS.md",
        "LICENSE",
        "CITATION.cff",
        "THIRD_PARTY_NOTICES.md",
    ]:
        if not (ROOT / required).is_file():
            errors.append(f"Missing {required}")
    for name in ["README.md", "AUTHORS.md", "CITATION.cff"]:
        text = (ROOT / name).read_text()
        if "QIUWEIHAO" not in text or "ShawYiranXiao" not in text:
            errors.append(f"{name}: missing collaborator attribution")
    summary = {"text_files_checked": count, "errors": errors}
    print(json.dumps(summary, indent=2))
    return bool(errors)


if __name__ == "__main__":
    sys.exit(check())
