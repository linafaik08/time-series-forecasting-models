# Advanced Time Series Forecasting with Nixtla

* Author: [Lina Faik](https://www.linkedin.com/in/lina-faik/)
* Creation date: May 2025
* Last update: May 2025

## Objective

This repository showcases advanced time series forecasting techniques using Nixtla’s libraries, with a focus on deep learning models such as TimeLLM. It includes traditional statistical models, transformer-based models (TimeGPT), performance benchmarking, and explainability components. All models are trained and evaluated on the M5 forecasting dataset.

For more context and insights, check out the companion blog post on [The AI Practitionner](https://aipractitioner.substack.com/).


## Project Description

This project provides a unified pipeline for exploring, training, and evaluating time series forecasting models using the M5 dataset. It integrates a wide range of models, from classical baselines to state-of-the-art LLM-based models, through the **Nixtla ecosystem**, including:

* **`statsforecast`** for statistical models (AutoARIMA, ETS, Theta)
* **`neuralforecast`** for deep learning models (NHITS, NBEATS)
* **`TimeLLM`** and **`TimeGPT`** via the Nixtla API
* **`mlforecast`** for LightGBM-based forecasts
* **Explainability** via SHAP and custom visualizations

## Code Structure

```
data/
├── calendar.csv
├── sales_train_validation.csv
├── sales_train_evaluation.csv
├── sell_prices.csv

notebooks/
├── 01_data_processing.ipynb              # Data loading & preprocessing
├── 02_model_training_stats.ipynb         # Statistical models (ARIMA, ETS)
├── 03_model_training_dl.ipynb            # Deep learning models (NHITS, NBEATS)
├── 04_model_training_timellm.ipynb       # TimeLLM (language-based model)
├── 05_model_training_timegpt.ipynb       # TimeGPT (transformer-based model)
├── 06_model_explainability.ipynb         # SHAP & insights on predictions
├── 07_perf_comparison.ipynb              # Final performance benchmarking

src/
├── data_processing.py                    # Data preparation logic
├── training.py                           # Model training functions
├── metrics.py                            # Custom metric implementations
├── viz.py                                # Visualization utilities

output/
├── [Generated forecasts, plots, metrics]

config.py                                 # Global paths and API keys
requirements.txt                          # Python dependencies
README.md
```

## How to Use This Repository?

### 1. Data

Download the **M5 Forecasting - Accuracy** dataset from Kaggle:[https://www.kaggle.com/competitions/m5-forecasting-accuracy/data](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

Place the following files in the `data/` directory:

* `calendar.csv`
* `sales_train_validation.csv`
* `sales_train_evaluation.csv`
* `sell_prices.csv`
* `sample_submission.csv` *(optional)*


### 2. Nixtla API Key (for TimeLLM & TimeGPT)

To run TimeLLM or TimeGPT models, open `config.py` and set your Nixtla API key:

```python
NIXTLA_API_KEY = "your_nixtla_api_key_here"
```

You can obtain a key from: [https://nixtla.io/](https://nixtla.io/)


### 3. Installation

Install all required packages with:

```bash
pip install -r requirements.txt
```

Main dependencies:

```
pandas
numpy
plotly
tqdm
matplotlib
nbformat
scikit-learn
shap
lightgbm
statsforecast
neuralforecast
mlforecast
transformers
nixtla
```

Python ≥3.11 is recommended for compatibility.

## Running the Project

Run the following notebooks in order:

1. `01_data_processing.ipynb` – Prepare the M5 dataset using `src/data_processing.py`
2. Train models:

   * `02_model_training_stats.ipynb` – Traditional models via `statsforecast`
   * `03_model_training_dl.ipynb` – Deep learning models with `neuralforecast`
   * `04_model_training_timellm.ipynb` – Run TimeLLM via Nixtla API
   * `05_model_training_timegpt.ipynb` – Run TimeGPT (transformer forecaster)
3. `06_model_explainability.ipynb` – Visualize and interpret forecasts (SHAP etc.)
4. `07_perf_comparison.ipynb` – Compare all model performances and draw conclusions


## Notes

* TimeGPT require an internet connection and valid API key.
* All generated outputs (forecasts, plots, comparisons) are stored in the `output/` directory.
* The code is modular and designed for easy extension to other datasets or models.
