#!/usr/bin/env bash
set -euo pipefail

echo "MAPPO-EDA Linux setup checks (no packages are installed by this script)."
echo "Run on Ubuntu, not on the macOS development machine."

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: this script is for the Ubuntu/Linux deployment only." >&2
  exit 2
fi

missing=0
check() {
  if command -v "$1" >/dev/null 2>&1; then
    printf 'OK   %-16s %s\n' "$1" "$($1 --version 2>/dev/null | head -n 1 || true)"
  else
    printf 'MISS %-16s\n' "$1"
    missing=1
  fi
}

check python3
check uv
check git
check make
check tclsh
check openroad
check yosys
check klayout

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  echo "MISS nvidia-smi       CUDA driver visibility is required for RTX 4090 training"
  missing=1
fi

if [[ -n "${ORFS_HOME:-}" && -d "$ORFS_HOME" ]]; then
  echo "OK   ORFS_HOME         $ORFS_HOME"
  if [[ -d "$ORFS_HOME/.git" ]]; then
    orfs_revision="$(git -C "$ORFS_HOME" rev-parse --short=7 HEAD 2>/dev/null || true)"
    if [[ "$orfs_revision" == "8b8d340" ]]; then
      echo "OK   ORFS revision     $orfs_revision"
    else
      echo "WARN ORFS revision     found $orfs_revision; expected 8b8d340"
      missing=1
    fi
    if [[ ! -f "$ORFS_HOME/flow/designs/nangate45/gcd/config.mk" ]]; then
      echo "MISS ORFS GCD config   $ORFS_HOME/flow/designs/nangate45/gcd/config.mk"
      missing=1
    else
      echo "OK   ORFS GCD config   nangate45/gcd"
    fi
  else
    echo "MISS ORFS checkout     $ORFS_HOME is not a git checkout"
    missing=1
  fi
else
  echo "MISS ORFS_HOME         set it to an OpenROAD-flow-scripts checkout"
  missing=1
fi

if command -v uv >/dev/null 2>&1; then
  uv sync --dev
else
  echo "Install uv from https://docs.astral.sh/uv/ before running uv sync."
fi

if [[ "$missing" -ne 0 ]]; then
  cat <<'EOF'

Some tools are missing. An administrator may install Ubuntu prerequisites with:
  sudo apt-get update
  sudo apt-get install -y build-essential git make python3 python3-venv tcl yosys klayout

Install OpenROAD/ORFS and NVIDIA CUDA drivers using their official Linux
instructions. This script deliberately does not run sudo or install packages.
Clone the pinned ORFS revision with:
  git clone --recursive https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git "$HOME/OpenROAD-flow-scripts"
  git -C "$HOME/OpenROAD-flow-scripts" checkout 8b8d340
  git -C "$HOME/OpenROAD-flow-scripts" submodule update --init --recursive
  export ORFS_HOME="$HOME/OpenROAD-flow-scripts"
EOF
  exit 2
fi

echo "Linux prerequisites are visible. Run scripts/openroad/verify_orfs_gcd.sh next."
