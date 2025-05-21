# Advanced Time Series Forecasting with Nixtla

* Author: [Lina Faik](https://www.linkedin.com/in/lina-faik/)
* Creation date: May 2025
* Last update: May 2025

## Objective

This repository showcases advanced time series forecasting techniques using Nixtla’s libraries, with a focus on deep learning models such as TimeLLM. It includes traditional statistical models, neural models, performance benchmarking, and explainability components. All models are trained and evaluated on the M5 forecasting dataset.

For more context and insights, check out the companion blog post on [The AI Practitionner](https://aipractitioner.substack.com/).

## Project Description

This project provides a unified pipeline for exploring, training, and evaluating time series forecasting models using the M5 dataset. It integrates a wide range of models, from classical baselines to state-of-the-art LLM-based models, through the **Nixtla ecosystem**, including:

* **`statsforecast`** for fast statistical models (AutoARIMA, ETS, Theta)
* **`neuralforecast`** for neural models like NHITS and NBEATS
* **`TimeLLM`**, a language-model-based forecaster via the Nixtla API
* **`mlforecast`** for gradient-boosted models using LightGBM
* **Explainability tools** to interpret model outputs (e.g., SHAP)

The goal is to provide robust model comparisons, practical usage patterns, and reproducible pipelines for real-world forecasting tasks.

### Code Structure

```
data/
├── calendar.csv
├── sales_train_validation.csv
├── sales_train_evaluation.csv
├── sell_prices.csv

notebooks/
├── 01_data_processing.ipynb             # Data preparation
├── 02_model_training_stats.ipynb        # Traditional statistical models
├── 03_model_training_dl.ipynb           # Deep learning models (e.g., NHITS, NBEATS)
├── 04_model_training_timellm.ipynb      # TimeLLM training and inference
├── 05_perf_comparison.ipynb             # Performance comparison across models

src/
├── data_processing.py                   # Data loading and formatting
├── training.py                          # Model training logic
├── metrics.py                           # Custom evaluation metrics
├── viz.py                               # Visualization utilities

output/
├── [Generated forecasts and plots]

config.py                                # Config file with Nixtla API key and paths
requirements.txt                         # Python dependencies
README.md
```

## How to Use This Repository?

### 1. Data

Please download the **M5 Forecasting - Accuracy** dataset from Kaggle:
[https://www.kaggle.com/competitions/m5-forecasting-accuracy/data](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

Place the following files in the `data/` directory:

* `calendar.csv`
* `sales_train_validation.csv`
* `sales_train_evaluation.csv`
* `sell_prices.csv`
* `sample_submission.csv`

### 2. Nixtla API Key (for TimeLLM only)

To use the **TimeLLM** model, set your Nixtla API key in `config.py`:

```python
NIXTLA_API_KEY = "your_nixtla_api_key_here"
```

You can obtain a key from: [https://nixtla.io/](https://nixtla.io/)

### 3. Requirements

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

Main dependencies include:

```
plotly
pandas
numpy
nbformat
tqdm
scikit-learn
nixtla
shap
matplotlib
neuralforecast
transformers
statsforecast
mlforecast
lightgbm
```

Make sure you're using Python ≥3.11 for full compatibility.

## Running the Project

1. Start with `01_data_processing.ipynb` to prepare the dataset.
2. Train and evaluate models:

   * `02_model_training_stats.ipynb`: Statistical models
   * `03_model_training_dl.ipynb`: Neural network models
   * `04_model_training_timellm.ipynb`: TimeLLM via Nixtla
3. Compare results in `05_perf_comparison.ipynb`.

## Notes

* TimeLLM calls require an active internet connection and a valid Nixtla API key.
* All forecast results and evaluation metrics are stored in the `output/` directory.
* This repository is modular and easy to extend with additional models or datasets.