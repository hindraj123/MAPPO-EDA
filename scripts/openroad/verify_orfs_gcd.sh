#!/usr/bin/env bash
set -euo pipefail

# Run the official ORFS Nangate45/GCD flow. This script intentionally does not
# copy, patch, or vendor technology files into MAPPO-EDA.

expected_revision="8b8d340"
orfs_home="${ORFS_HOME:-}"
if [[ -z "$orfs_home" || ! -d "$orfs_home/flow" ]]; then
  echo "ERROR: set ORFS_HOME to an ORFS checkout containing flow/" >&2
  exit 2
fi
if ! git -C "$orfs_home" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: ORFS_HOME must be a git checkout so the revision can be verified" >&2
  exit 2
fi

actual_revision="$(git -C "$orfs_home" rev-parse --short=7 HEAD)"
if [[ "$actual_revision" != "$expected_revision" ]]; then
  echo "ERROR: ORFS revision $actual_revision does not match $expected_revision" >&2
  exit 2
fi

config="$orfs_home/flow/designs/nangate45/gcd/config.mk"
if [[ ! -f "$config" ]]; then
  echo "ERROR: official ORFS GCD configuration is missing: $config" >&2
  exit 2
fi

echo "ORFS revision: $actual_revision"
echo "Running official Nangate45/GCD configuration from: $config"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$repo_root/artifacts"
run_dir="$(mktemp -d "$repo_root/artifacts/orfs-gcd-XXXXXXXX")"
marker="$run_dir/started"
touch "$marker"
{
  echo "ORFS revision: $(git -C "$orfs_home" rev-parse HEAD)"
  git -C "$orfs_home" status --short
  for tool in openroad yosys klayout; do
    if command -v "$tool" >/dev/null 2>&1; then
      printf '%s: ' "$tool"
      case "$tool" in
        yosys) "$tool" -V 2>&1 | head -n 1 || true ;;
        klayout) "$tool" -v 2>&1 | head -n 1 || true ;;
        *) "$tool" -version 2>&1 | head -n 1 || true ;;
      esac
    else
      printf '%s: not on PATH\n' "$tool"
    fi
  done
} > "$run_dir/toolchain.txt"
echo "Run logs: $run_dir"
cd "$orfs_home/flow"
make DESIGN_CONFIG=./designs/nangate45/gcd/config.mk 2>&1 | tee "$run_dir/flow.log"

gds="$orfs_home/flow/results/nangate45/gcd/base/6_final.gds"
if [[ ! -s "$gds" ]]; then
  echo "ERROR: expected GDSII artifact was not produced: $gds" >&2
  exit 1
fi
if [[ ! "$gds" -nt "$marker" ]]; then
  echo "ERROR: GDSII predates this run; existing output alone does not verify a fresh flow: $gds" >&2
  exit 1
fi
sha256sum "$gds" > "$run_dir/gds.sha256"
echo "GDSII verified present: $gds"
