# eval-support-diag

Diagnostics for evaluation-support instability in spatial crop classification.

Companion code for:

> Distinguishing evaluation-support instability from intrinsic regional difficulty in spatial crop classification.
> Ecological Informatics (submitted).

## What it does

- Precision/recall decomposition of a regional F1 gap into base-rate, aggregation-weight, and classifier-rate components.
- Shapley symmetric decomposition (removes order sensitivity).
- Parcel-level cluster bootstrap for component-share uncertainty.
- Confusion-matrix simulation aligned with the matched-support experiment and validating component recovery under known ground truth.

## Install

    pip install -r requirements.txt
    pip install -e .

## Quick start

    python scripts/run_pr_decomposition.py --base-dir outputs --data-dir data/features --out outputs
    python scripts/run_decomp_order.py      --base-dir outputs --data-dir data/features --out outputs
    python scripts/run_sim_baserate.py      --rates outputs/pr_rates_by_region.csv --out outputs

## Outputs

| File | Used for |
|---|---|
| outputs/pr_rates_by_region.csv | per-region per-class rates |
| outputs/pr_decomposition.csv | three-step decomposition |
| outputs/decomp_order_bootstrap_summary.csv | main-text Shapley shares |
| outputs/decomp_order_bootstrap_by_scale.csv | supplementary per-scale |
| outputs/sim_partA_table7.csv | Table 7 alignment |
| outputs/sim_partB_scenarios.csv | simulation scenarios |

## Data

The labelled field-survey samples cannot be released. See data/README.md
for the expected directory layout. Sentinel-1 and Sentinel-2 imagery is
publicly available from the Copernicus Data Space Ecosystem.

## License

MIT - see LICENSE.
