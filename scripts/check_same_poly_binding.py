#!/usr/bin/env python3
"""CI lint: reject discarded verify_same_poly results over uncommitted data.

verify_same_poly returns the merged claim (r, f(r)); the caller MUST
authenticate it against f's commitment (or f must be verifier-recomputed).
Every `.is_none()` discard in production code must carry a `LINT-ALLOW`
justification comment.
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = (root / "crates/zkie-ops/src/compose.rs").read_text()
prod = src.split("#[cfg(test)]", 1)[0]
lines = prod.splitlines()

hits = []
for i, line in enumerate(lines):
    if "verify_same_poly" not in line:
        continue
    window = lines[i:i + 12]
    if not any(".is_none()" in x for x in window):
        continue
    marked = any("LINT-ALLOW" in x for x in lines[max(0, i - 2):i + 12])
    if not marked:
        hits.append((i + 1, line.strip()))

if hits:
    for ln, text in hits:
        print(f"{ln}: unauthenticated verify_same_poly discard: {text}", file=sys.stderr)
    sys.exit(1)
print("same_poly binding lint: OK")
