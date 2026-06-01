# Legacy scripts (exploratory)

**Canonical pipeline:**

| Stage | Directory |
|-------|-----------|
| A | [stage_a/](../stage_a/) |
| B | [stage_b/](../stage_b/) |
| C | [stage_c/](../stage_c/) |

This folder contains historical and exploratory utilities (DiffAE, flow matching, MIL, orientation sweeps, etc.). They are **not** required to reproduce the thesis Plain/MONAI → spatial v2 → repeated eval path.

For new work, use [stage_c/run/run_paired_ensemble.sh](../stage_c/run/run_paired_ensemble.sh) instead of duplicating wrappers here.
