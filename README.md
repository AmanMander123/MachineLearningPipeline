# Reproducible rental price training pipeline

This Udacity ML DevOps project estimates short-term rental listing prices in New York City. It treats cleaning, validation, and training as repeatable stages, so a new data snapshot can be checked before it reaches the model.

## What I built

I extended the [Udacity starter](https://github.com/udacity/nd0821-c2-build-model-workflow-starter) with exploratory analysis, cleaning, data checks, random forest training, and configuration for the selected hyperparameters. The workflow uses MLflow to run stages, Hydra for settings, and Weights & Biases for experiment tracking and artifact lineage.

The pipeline validates column order, borough coverage, geographic bounds, row count, and prices. It also compares the borough distribution with a reference artifact before training. Keeping the reference check separate makes it possible to inspect a new snapshot without silently changing what counts as normal.

## Run the local data checks

From the repository root, install the local dependencies in a Python 3.11 environment:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install pandas==2.2.3 scipy==1.11.4
python - <<'PY'
import runpy

import pandas as pd

checks = runpy.run_path("src/data_check/test_data.py")
sample = pd.read_csv("src/basic_cleaning/clean_sample.csv")
for name in (
    "test_column_names",
    "test_neighborhood_names",
    "test_proper_boundaries",
    "test_row_count",
):
    checks[name](sample)
    print(f"{name}: passed")
checks["test_price_range"](sample, 10, 350)
print(f"test_price_range: passed ({len(sample):,} rows)")
PY
```

All five checks passed for the 19,001 rows in the checked-in sample. This walkthrough covers the existing checks that use a single dataset. The drift comparison needs a separate reference artifact, and the full tracking and training workflow needs the historical MLflow/Conda environment and Weights & Biases artifacts.

## Project files

- [Pipeline configuration](config.yaml) specifies cleaning thresholds, data splits, and training settings.
- [Data checks](src/data_check/test_data.py) contain the schema and distribution checks.
- [Training stage](src/train_random_forest/run.py) fits the regressor and records model artifacts.
- [Original experiment workspace](https://wandb.ai/amanmander123/nyc_airbnb) is a historical reference.
