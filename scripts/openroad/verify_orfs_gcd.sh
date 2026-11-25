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
if [[ ! -d "$orfs_home/.git" ]]; then
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
cd "$orfs_home/flow"
make DESIGN_CONFIG=./designs/nangate45/gcd/config.mk

gds="$orfs_home/flow/results/nangate45/gcd/base/6_final.gds"
if [[ ! -s "$gds" ]]; then
  echo "ERROR: expected GDSII artifact was not produced: $gds" >&2
  exit 1
fi
echo "GDSII verified present: $gds"
