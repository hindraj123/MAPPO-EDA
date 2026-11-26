# MAPPO-EDA

MAPPO-EDA is a cooperative multi-agent reinforcement-learning framework for
RTL-to-GDSII physical-design optimization with OpenROAD.

## Research scope

The system contains four cooperative agents (floorplanning, placement,
clock-tree synthesis, and routing), decentralized actors, a centralized MAPPO
critic, multi-view GNN/CNN/MLP embeddings, stage-aware masks, parallel CPU
rollout workers, and RTX 4090 training. MAPPO is implemented from scratch in
PyTorch and uses only fresh on-policy rollouts.

The project deliberately does not include a persistent trajectory database,
offline RL, historical dataset management, or LLM optimization agents.

## Lightweight experiment artifacts

Each run writes its configuration, random seeds, toolchain versions, JSONL/CSV
metrics and events, model checkpoints, failure logs, and generated GDSII files
under `artifacts/<run-id>/`. The on-policy rollout buffer is an in-memory
training component and is not a historical data store.

## Real-training telemetry

Real training uses W&B online by default with project `MAPPO-EDA`, run name
`gcd-nangate45-mappo-4env`, and group `real-openroad-training`. The SDK uses
its existing local authentication; credentials must not be placed in project
configuration. W&B is initialized only in the training process. The optional
`telemetry` dependency group declares `wandb`, `psutil`, and `nvidia-ml-py`;
the existing Python environment is not modified by the training script. If
the W&B SDK is absent, real training fails with an explicit error. Set
`wandb_enabled: false` only when online tracking is intentionally disabled.

The run records stage rewards and PPO diagnostics, available system readings,
and physical metrics actually emitted by the configured OpenROAD extraction
script. Missing power, congestion, DRC, or other measurements are omitted, not
estimated. GPU utilization is included when NVML is available and the CUDA
device UUID can be matched. `baseline_metrics` and `reward_directions` in the
experiment JSON enable per-metric baseline-relative improvements; each baseline
value must come from a comparable real default flow. Completed episodes are
counted across updates, with a checkpoint after crossing each 50-episode
boundary (at the next completed optimization update). The W&B run ID is stored
in the checkpoint and a resumed run must reconnect to that same ID.

To retain a best GDSII, the experiment must specify `gds_filename` relative
to each episode's isolated working directory and a `best_metric` with a
corresponding direction in `reward_directions`. The final stage must emit both
that metric and `drc_violations`; only completed episodes reporting zero DRC
violations and an existing, nonempty GDSII file are eligible. The current GCD
stage-action integration and its full metric extraction remain unverified.

No data flywheel infrastructure is included: there is no persistent trajectory
database, offline RL, or historical dataset manager.

## Development versus Linux training

The macOS machine is a portable development host. Do not install Docker,
OpenROAD, or Yosys there. The Ubuntu RTX 4090 workstation is the target for
the real OpenROAD-flow-scripts GCD run, integration tests, and CUDA training.

The scripts provide a clearly labeled synthetic CPU harness for testing the
existing RL core. Synthetic rewards and observations are not physical-design
measurements. Real OpenROAD verification is **pending** until the Linux
workstation executes and records the complete RTL-to-GDSII flow.

## Commands

For a Mac terminal viewer and evaluated GDS downloads from the Linux desktop,
see [Mac remote setup](docs/mac-remote.md). Connection settings stay in an ignored
local config; a portable example is provided in `configs/remote.example.json`.

```bash
uv sync --dev
./scripts/smoke_test.sh
```

On Ubuntu, inspect prerequisites without automatic privileged changes:

```bash
./scripts/setup_linux.sh
```

Run the development harness:

```bash
PYTHONPATH=src uv run python train.py --synthetic --device cpu
uv run python evaluate.py artifacts/synthetic-development/checkpoint-00002.pt
```

CUDA is selected by configuration/CLI on the Ubuntu workstation; no GPU
identifier is hardcoded:

```bash
PYTHONPATH=src uv run python train.py --synthetic --device cuda
```

`configs/openroad.yaml` documents the pending Linux toolchain integration. The
OpenROAD wrapper and extraction code are present under `src/`; this repository
does not claim real-tool verification until that workstation run is completed.

## Official ORFS Nangate45/GCD verification

The Linux integration is pinned to OpenROAD-flow-scripts commit `8b8d340`.
This revision is used with the official ORFS configuration at
`flow/designs/nangate45/gcd/config.mk`; ORFS supplies the Nangate45 technology
files and flow inputs. No LEF, Liberty, GDS, or layer-map files are vendored in
this repository. The expected ORFS artifact is
`flow/results/nangate45/gcd/base/6_final.gds`.

On Ubuntu, after installing/configuring the external toolchain:

```bash
export ORFS_HOME="$HOME/OpenROAD-flow-scripts"
git -C "$ORFS_HOME" checkout 8b8d340
git -C "$ORFS_HOME" submodule update --init --recursive
./scripts/setup_linux.sh
./scripts/openroad/verify_orfs_gcd.sh
```

`setup_linux.sh` only checks dependencies and documents administrator actions;
it does not run `sudo` and must not be used to install EDA tools on macOS.
The verification script runs the official ORFS `make` flow and checks that the
GDSII artifact exists. Until those commands succeed on Ubuntu, real OpenROAD
verification remains **pending**.

## Phase 1 status

Phase 1 provides the OpenROAD command wrapper, structured design-state schema,
lightweight run logging, and GCD benchmark scaffolding. OpenROAD and ORFS are
external prerequisites and are intentionally not vendored into this repository.

```bash
uv run pytest
```
