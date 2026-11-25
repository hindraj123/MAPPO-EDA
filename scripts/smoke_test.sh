#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

# Keep uv's cache portable and outside restricted host cache locations.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/mappo-eda-uv-cache}"

if command -v uv >/dev/null 2>&1; then
  runner=(uv run)
else
  runner=(python3)
fi

"${runner[@]}" - <<'PY'
import ast
from pathlib import Path

for path in [Path("train.py"), Path("evaluate.py"), *Path("scripts").glob("*.py")]:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("Python syntax checks passed")
PY
"${runner[@]}" -m pytest

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/mappo-eda-smoke.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT
PYTHONPATH=src "${runner[@]}" train.py --synthetic --config configs/training.yaml --run-dir "$tmp_dir/run" --updates 1 --steps 2 --device cpu
test -f "$tmp_dir/run/checkpoint-00001.pt"
PYTHONPATH=src "${runner[@]}" evaluate.py "$tmp_dir/run/checkpoint-00001.pt" >/dev/null
echo "Synthetic CPU smoke test passed; it does not verify real OpenROAD training."
