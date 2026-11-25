# MAPPO-EDA agent instructions

## Scope

MAPPO-EDA is a cooperative, four-agent physical-design research framework:

- floorplanning
- placement
- clock-tree synthesis (CTS)
- routing

The learning algorithm is MAPPO implemented in PyTorch with centralized
training and decentralized execution. The actor observation path uses GNN,
CNN, and MLP views followed by feature fusion. The centralized critic consumes
global state. Rollouts are strictly on-policy and held in memory only.

## Development and training machines

- macOS is for portable development, unit tests, and synthetic smoke tests.
- Ubuntu with an RTX 4090 is the target for OpenROAD integration tests and CUDA
  training.
- Do not install Docker, OpenROAD, or Yosys on the macOS development machine.
- Do not hardcode paths, GPU identifiers, or machine-specific tool locations.

## Evidence policy

Synthetic fixtures are permitted only in unit tests and local smoke tests and
must be labeled synthetic. Never present synthetic values as physical-design
measurements. Real OpenROAD verification is pending until the Linux workstation
executes the GCD RTL-to-GDSII flow.

## Artifact policy

Use lightweight per-run artifacts only: configuration snapshots, seeds,
toolchain-version reports, JSONL/CSV events, failure logs, checkpoints, and
generated GDSII files. Do not add persistent trajectory databases, offline RL,
or historical dataset management.

## Change boundaries

Keep algorithmic changes in `src/mappo_eda/rl/` and environment changes in the
environment package. Top-level scripts should orchestrate those interfaces and
must not fabricate OpenROAD metrics.

## Validation

Run `scripts/smoke_test.sh` on development machines. On Ubuntu, also run the
real OpenROAD benchmark and record its tool versions and results before calling
the integration verified.
