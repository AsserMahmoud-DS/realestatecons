# Interpretable House Price Prediction

This repository contains a machine learning course project for predicting house
sale prices with interpretable modeling, comparative evaluation, uncertainty
estimation, and an eventual Streamlit deployment interface.

The project uses the Kaggle
["House Prices: Advanced Regression Techniques"](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
dataset. The current work focuses on exploratory analysis, data cleaning,
feature engineering, baseline regression models, probabilistic modeling, and a
TensorFlow neural-network baseline.

## Project Aim

The goal is to build a clear, reproducible house price prediction workflow that
does more than produce a single score. The project is designed to:

- clean and explore the Kaggle housing dataset;
- engineer useful real-estate features such as total square footage, bathroom
  totals, outdoor area, and property age;
- compare classical regression models and a neural-network baseline;
- evaluate predictions with RMSE, MAE, and R2;
- save generated figures, submissions, and metrics in organized project folders;
- provide interpretable outputs such as coefficients, feature importance,
  residual analysis, and predicted-vs-actual plots;
- add uncertainty estimates or prediction intervals for model outputs;
- later expose the trained pipeline through a Streamlit app.

## Repository Structure

```text
.
├── app/                         # Streamlit app package placeholder
├── data/
│   ├── raw/                     # Kaggle train/test files
│   └── processed/               # Processed data outputs
├── models/                      # Saved trained models
├── notebooks/                   # EDA, modeling, and experiment notebooks
├── reports/
│   ├── figures/                 # Generated visualizations
│   └── submissions/             # Kaggle-style prediction submissions
├── src/
│   ├── data/                    # Data cleaning helpers
│   ├── features/                # Feature engineering helpers
│   └── models/                  # Training scripts
├── tests/                       # Pytest tests
├── context.md                   # Original project brief
└── requirements.txt
```

## Dataset Setup

Download the Kaggle competition data and place the files here:

```text
data/raw/train.csv
data/raw/test.csv
```

The target column is `SalePrice`. Raw Kaggle data may be excluded from version
control depending on licensing and file-size constraints.

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebook Workflow

The notebooks are intended for step-by-step experimentation and should be run
from the repository root so relative paths resolve correctly.

Open them with Jupyter, JupyterLab, or VS Code after installing the project
requirements.

Recommended order:

1. `notebooks/eda_cleaning.ipynb` - initial EDA and missing-value cleaning.
2. `notebooks/eda_features.ipynb` - feature exploration and engineered features.
3. `notebooks/ml_models_pipelines.ipynb` - classical ML models and submissions.
4. `notebooks/probabilistic_pipelines.ipynb` - uncertainty/probabilistic models.
5. `notebooks/neural_network_pipeline.ipynb` - neural-network experiments.

Generated plots are saved under `reports/figures/` using topic-specific
subfolders such as `correlation/`, `missing_values/`, `pairplot/`, and
`neural_network/`.

## Current Usage

Reusable cleaning and feature engineering logic lives in `src/`:

- `src/data/preprocess.py`
- `src/features/features.py`

The currently available command-line training script is the TensorFlow neural
network pipeline. Run it from the repository root:

```bash
python -m src.models.train_neural_network_tensorflow
```

This script:

- loads `data/raw/train.csv` and `data/raw/test.csv`;
- applies cleaning and feature engineering helpers;
- creates a train/validation split;
- compares 1-, 2-, and 3-hidden-layer neural-network architectures;
- grid-searches learning rate and batch size;
- saves validation metrics to `reports/metrics/`;
- saves the best loss curve to
  `reports/figures/neural_network/tensorflow_best_loss_curve.png`;
- saves the best Kaggle-style submission to
  `reports/submissions/neural_net_tensorflow_best.csv`.

Classical model experiments and existing submission outputs are currently
managed through the notebooks.

## Current Project Status

Implemented:

- project folder structure;
- raw data location convention;
- cleaning helpers for train/test data;
- feature engineering helpers;
- EDA and modeling notebooks;
- generated EDA figures;
- classical-model submission outputs from notebook experiments;
- TensorFlow neural-network training script and submission export.

In progress / planned:

- consolidated `src/train.py` for classical model comparison;
- saved scikit-learn preprocessing/model pipelines in `models/`;
- pytest coverage for important transformations;
- Streamlit prediction app in `app/streamlit_app.py`;
- app-level uncertainty display and feature explanation outputs.

## Validation

As the project matures, the intended validation commands are:

```bash
python -m pytest
python src/train.py
streamlit run app/streamlit_app.py
```

At the current stage, `src/train.py` and `app/streamlit_app.py` are not yet
implemented. The available checks are:

```bash
python -m pytest
python -m src.models.train_neural_network_tensorflow
```

## Outputs

Current generated artifacts include:

- EDA visualizations in `reports/figures/`;
- neural-network loss curve in `reports/figures/neural_network/`;
- Kaggle-style prediction files in `reports/submissions/`.
