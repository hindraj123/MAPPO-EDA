# GCD benchmark

This directory contains the small, synthesizable RTL fixture used to exercise
the OpenROAD integration. It is source input, not a measured result: no area,
timing, congestion, power, or routing numbers are checked into the repository.

The direct OpenROAD flow in `scripts/openroad/gcd_flow.tcl` requires a Linux
deployment to provide a compatible technology LEF, Liberty, and GDS layer-map.
The paths are supplied at runtime through environment variables so the
repository remains portable:

```text
MAPPO_EDA_GCD_RTL       path to rtl/gcd.v
MAPPO_EDA_LEF           technology and standard-cell LEF (or a Tcl-readable LEF list)
MAPPO_EDA_LIBERTY       standard-cell Liberty
MAPPO_EDA_SDC           path to constraints/gcd.sdc
MAPPO_EDA_DIE_AREA      four numbers: x_min y_min x_max y_max
MAPPO_EDA_CORE_AREA     four numbers: x_min y_min x_max y_max
MAPPO_EDA_GDS           output GDSII path
MAPPO_EDA_GDS_MAP       GDS layer map path
MAPPO_EDA_OUTPUT_DIR    output directory for DEF/ODB artifacts
```

The technology files are intentionally not included. A GDSII artifact is valid
only after the real Linux OpenROAD run completes successfully.

## Official Linux integration

For the real benchmark, use the official OpenROAD-flow-scripts checkout pinned
to commit `8b8d340`, not this small local RTL fixture as a replacement for ORFS
inputs. ORFS's `flow/designs/nangate45/gcd/config.mk` selects the official
Nangate45 platform and GCD flow. Set `ORFS_HOME` to that checkout and run:

```bash
./scripts/openroad/verify_orfs_gcd.sh
```

from the repository root (or use the equivalent root-relative command). The
expected artifact is `$ORFS_HOME/flow/results/nangate45/gcd/base/6_final.gds`.
Real execution and all physical-design measurements remain pending until the
Ubuntu workstation runs this flow successfully.
