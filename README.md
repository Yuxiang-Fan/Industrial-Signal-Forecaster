# Industrial Signal Forecaster (CNN-LSTM)

This repository contains my solution for the **2025 "Supcon Cup" (中控杯) Industrial AI Innovation Challenge** - Industrial Time-Series Forecasting Track. 

The project focuses on predicting trends in high-dimensional industrial sensor data by combining statistical feature engineering with deep learning.

## 🛠️ Implementation Highlights

Instead of using a vanilla model, I implemented several practical strategies to handle the complexities of real-world industrial datasets:

* **Lag-Correlation & Feature Selection**: To account for the delayed response in industrial processes, I analyzed the Spearman rank correlation across multiple lag orders. Redundant or highly collinear features were removed to reduce noise and improve model convergence.
* **Memory-Efficient Scaling (OOM Prevention)**: Facing a massive dataset that could lead to Out-Of-Memory (OOM) errors, I calculated global Min-Max statistics (`stats.xls`) offline. These are applied on-the-fly within the PyTorch `DataLoader`, ensuring zero data leakage while keeping a small memory footprint.
* **CNN-LSTM Hybrid Architecture**: I designed a simple yet effective network using 1D-CNN layers to extract local temporal patterns, followed by LSTM layers to capture long-term sequential dependencies.
* **Strict Temporal Validation**: All data splitting and windowing follow a strict chronological order. This ensures that no "future data" is leaked into the training phase, maintaining the integrity of the forecasting results.

## 📁 Project Structure

```text
Industrial-Signal-Forecaster/
├── data/                    # Dataset directory
│   ├── sample_data.xls      # 100-row sample to demonstrate the data schema
│   └── README.md            # Instructions on the original competition data
├── features/                # Feature engineering artifacts
│   ├── selected_cols_123.xls # Selected lag features for Signal 123
│   ├── selected_cols_124.xls # Selected lag features for Signal 124
│   └── stats.xls            # Pre-computed statistics for online scaling
├── logs/                    # Execution logs
│   └── lag_correlation.log  # Detailed logs of the feature selection process
├── src/                     # Core source code
│   ├── feature_engineering.py # Parallel lag-correlation & collinearity processing
│   ├── models.py            # CNN-LSTM network architecture definition
│   ├── train.py             # Training loop, validation, and checkpointing
│   └── predict.py           # Inference script for generating test submissions
├── notebooks/               # Experimental records
│   └── EDA_and_Training.ipynb # Exploratory Data Analysis & visual training logs
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
