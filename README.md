# Industrial Signal Forecaster (CNN-LSTM)

This project provides a specialized deep learning implementation for predicting trends in high-dimensional industrial sensor data. Developed for the **2025 "Supcon Cup" (中控杯) Industrial AI Innovation Challenge**, the repository demonstrates a practical approach to handling temporal dependencies and large-scale data constraints in industrial environments.

## 🛠️ Key Implementation Details

The solution integrates statistical feature engineering with a hybrid neural network architecture to address specific industrial forecasting challenges:

* **Lag-Correlation Analysis**: Computes Spearman rank correlation across multiple lag orders to identify delayed responses in sensor covariates. Redundant or highly collinear features are removed to streamline model input and reduce noise.
* **Memory-Optimized Data Pipeline**: To prevent Out-Of-Memory (OOM) errors during the training of massive datasets, the pipeline utilizes offline pre-computed statistics (`stats.xls`). Scaling is performed on-the-fly within the PyTorch `DataLoader`, ensuring memory efficiency and zero data leakage.
* **Hybrid CNN-LSTM Architecture**: Features 1D-CNN layers for local spatial-temporal feature extraction, followed by LSTM layers to capture long-term sequential dependencies.
* **Sequential Validation**: Implements a strict sliding-window mechanism for data preparation to ensure no future information is leaked during the training or validation phases.

## 📁 Repository Structure

```text
Industrial-Signal-Forecaster/
├── data/                    # Dataset storage and data schema samples
├── features/                # Artifacts from feature engineering (selected columns, stats)
├── logs/                    # Training and correlation analysis logs
├── src/                     # Core source code (models, training, and prediction)
├── notebooks/               # Exploratory Data Analysis (EDA) and experimental logs
├── requirements.txt         # Environment dependencies
└── README.md                # Documentation
