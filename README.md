# Industrial Signal Forecaster (CNN-LSTM)

A robust, end-to-end deep learning pipeline designed for high-dimensional industrial time-series forecasting. This project was developed for the **2025 "中控杯" (Supcon Cup) Industrial AI Innovation Challenge** - Industrial Time-Series Forecasting Track.

By combining rigorous statistical feature engineering with a hybrid CNN-LSTM neural network architecture, this project effectively captures both local temporal dependencies and long-term trends in complex industrial sensor data.

## ✨ Core Engineering Highlights

* **Lag-Correlation Analysis & Collinearity Removal**: Automatically evaluates the Spearman rank correlation across multiple lag orders for covariates. Highly correlated redundant features are dynamically removed to maximize information entropy and reduce noise.
* **Offline Min-Max Scaling**: To prevent Out-Of-Memory (OOM) errors common with massive industrial datasets, the pipeline pre-computes global min/max statistics (`stats.xls`). These constants are then dynamically applied during the PyTorch `DataLoader` phase (On-the-fly scaling), ensuring zero data leakage and high memory efficiency.
* **Hybrid CNN-LSTM Architecture**: Utilizes 1D-CNN layers for efficient local feature extraction across the time window, followed by LSTM layers to capture sequential dependencies.
* **Strict Temporal Windowing**: Implements a sliding-window mechanism for sequence generation, strictly preventing future data leakage into historical training batches.

## 📁 Project Structure

```text
Industrial-Signal-Forecaster/
├── data/                            # Dataset directory
│   ├── sample_data.xls              # A 100-row sample to demonstrate the data schema
│   └── README.md                    # Instructions on original competition data
├── features/                        # Decoupled feature engineering artifacts
│   ├── selected_cols_123.xls        # High-correlation lag features for Signal 123
│   ├── selected_cols_124.xls        # High-correlation lag features for Signal 124
│   └── stats.xls                       # Offline scaling statistics to prevent OOM
├── logs/                            # Execution logs
│   └── lag_correlation.log          # Detailed logs of the feature selection process
├── src/                             # Core source code
│   ├── feature_engineering.py       # Parallel lag-correlation & collinearity processing
│   ├── models.py                    # CNN-LSTM network architecture definition
│   ├── train.py                     # Training loop, validation, and checkpointing
│   └── predict.py                   # Inference script for generating test submissions
├── notebooks/                       # Experimental notebooks
│   └── EDA_and_Training.ipynb       # Exploratory Data Analysis and visual training logs
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
