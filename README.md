# 📡 5G QoS Predictive Modeler & Explainability (XAI) Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-15B8EC?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-FF6F00?style=flat-square)](https://shap.readthedocs.io/)
[![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat-square&logo=mysql&logoColor=white)](https://www.mysql.com/)
[![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI-2088FF?style=flat-square&logo=githubactions&logoColor=white)](.github/workflows/clean.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

An end-to-end Machine Learning and Explainable AI (XAI) pipeline designed to forecast **Quality of Service (QoS)** metrics and **Resource Allocation** in 5G cellular networks. The system benchmarks **XGBoost**, **Random Forest**, and **MLP** regressors while leveraging **SHAP (SHapley Additive exPlanations)** to interpret model predictions and satisfy telecommunication SLA transparency requirements.

---

## 🌟 Key Features

- **5G Telemetry Ingestion & Exploration:** Automated loading and type-casting of QoS network slices (application type, signal strength in dBm, latency in msec, required vs. allocated bandwidth in Mbps, resource allocation).
- **Supervised Regression Benchmarking:** Comparative training of **XGBoost Regressor**, **Random Forest Regressor**, and **MLP Regressor** with cross-validation and hyperparameter optimization.
- **Explainable AI (XAI) with SHAP:** Feature attribution via SHAP Summary Plots, Bar Charts, and Waterfall plots to unmask the impact of signal strength and latency on resource allocation decisions.
- **Persistent Telemetry & Metric Storage:** Direct MySQL integration saving performance evaluations ($R^2$, RMSE, MSE, MAPE) and run artifacts into relational tables.
- **Automated Visualization Engine:** Standalone visualization scripts generating high-resolution comparative diagnostic histograms and distribution curves.
- **Production-Ready Engineering Standards:** Configured with Pre-commit hooks (`flake8`, `black`, `isort`), `tox` environments, and GitHub Actions CI workflow.

---

## 📊 Model Evaluation & Explainability

### 1. Performance Benchmark (XGBoost vs. Random Forest)
Both models are quantitatively evaluated on held-out test splits across multiple regression metrics:

![Model Performance Comparison](graphs/model_performance_comparison_FINAL.png)

### 2. SHAP Feature Attribution (Interpretability)
SHAP summary plots identify the exact contribution and magnitude of each network feature towards 5G resource allocation decisions:

| Feature Importance (XGBoost) | Feature Importance (Random Forest) |
| :---: | :---: |
| ![SHAP XGBoost](graphs/shap_summary_plot_XGBoost_FINAL.png) | ![SHAP Random Forest](graphs/shap_summary_plot_Random Forest_FINAL.png) |

---

## 🏗️ Project Architecture & Directory Layout

```plaintext
5g-qos-predictive-modeler/
├── .github/workflows/      # Automated CI workflow (clean.yml)
├── config/                 # Configuration files & parameters
├── data/                   # 5G QoS dataset CSV & schema storage
├── docs/                   # Technical reports & documentation
├── graphs/                 # Exported SHAP plots, metrics, & histograms
├── helpers/                # Reusable utility modules
├── scripts/
│   ├── main.py             # Pipeline execution (data, training, SHAP, DB)
│   ├── Data Visualization.py # Graph generation from stored DB runs
├── .env-template           # Template for environment credentials
├── .gitignore              # Production-grade Python gitignore
├── .pre-commit-config.yaml # Linting & formatting configurations
├── Pipfile                 # Pipenv dependency management
├── requirements.txt        # Standard pip dependencies
├── setup.py                # Package setup script
└── tox.ini                 # Multi-environment testing configuration
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Mamel11/5g-qos-predictive-modeler.git
cd 5g-qos-predictive-modeler
```

### 2. Set Up Virtual Environment
```bash
# Using standard Python venv
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Database Configuration (Optional)
Configure your MySQL database parameters or review `.env-template`:
```bash
cp .env-template .env
```

### 5. Run the Pipeline
```bash
# Execute training, evaluation, SHAP analysis & database logging
python scripts/main.py

# Generate comparison plots from stored runs
python "scripts/Data Visualization.py"
```

All generated plots will be automatically exported to the `graphs/` directory.

---

## 🛠️ Code Quality & Pre-commit
To enforce clean code practices and PEP8 styling before committing:
```bash
pre-commit install
pre-commit run --all-files
```

---

## 📜 Authors & Acknowledgments

- **Emmanouela Xenou** ([@Mamel11](https://github.com/Mamel11)) - *Computer Engineering & Informatics (CEID), University of Patras*
- **Stefanos Metzidakis** - *Project Co-author*

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
