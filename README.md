# T2-weighted MRI models vs lesion volume for muscle-invasive bladder cancer — analysis code

Reproducible analysis code for a leave-one-center-out (LOCO) audit of T2-weighted MRI
radiomics and deep-learning models for muscle-invasive bladder cancer (MIBC), with an
evaluation-design (data-leakage / case-mix) audit.

This repository is released **anonymously** to support double-blind peer review.

## What is here

- `src/` — analysis scripts, run in numeric order (`s00`–`s22`) plus `common.py`.
  Each script's docstring states its inputs and outputs.
- `results/` — result tables (`table*.csv`, `posthoc_*.csv`, `table1_literature_extraction.csv`),
  the final figures (`fig_*.png` / `.pdf`) and run logs.
- `SAP.md` — the statistical analysis plan (version 1.0, 30 September 2026), fixed before any
  image model was trained.
- `MANIFEST.csv` — SHA-256 checksums of the released files.

## Data (not redistributed here)

The analysis uses the public **FedBCa** multicenter bladder-cancer MRI dataset
(Zenodo record 10409145). Images and masks are **not** redistributed in this repository.
To reproduce from raw data, download FedBCa from Zenodo and place the four centers as
`Center1/ … Center4/` next to `src/`, then run the scripts in order starting with `s00_index.py`.

Intermediate derived data (preprocessed volumes, crops, model checkpoints and per-lesion
predictions) are produced under a local `work/` directory by the scripts and are not shared.

## Requirements

Python 3.10; PyTorch 2.12, MONAI 1.6, PyRadiomics 3.0.1, SimpleITK 2.5, scikit-learn 1.6,
statsmodels 0.14, pandas 2.x, timm 0.9, transformers 4.x. A CUDA GPU is needed for the
deep-learning arms.

## License

Code released under the MIT License (see `LICENSE`). The FedBCa dataset is governed by its
own license on Zenodo.
