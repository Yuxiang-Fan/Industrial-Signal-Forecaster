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
```

---

# 工业信号预测器 (CNN-LSTM)

本项目提供了一个专门用于高维工业传感器数据趋势预测的深度学习实现方案。该项目是为 **2025“中控杯”工业 AI 创新挑战赛**开发的，展示了在工业环境下处理时间依赖性和大规模数据限制的实用方法。

## 🛠️ 核心实现细节

该方案将统计特征工程与混合神经网络架构相结合，以应对特定的工业预测挑战：

* **滞后相关性分析**：计算多个滞后阶数下的 Spearman 秩相关系数，以识别传感器协变量中的延迟响应。通过移除冗余或高度共线性的特征，精简模型输入并降低噪声。
* **内存优化数据流水线**：为防止在大规模数据集训练过程中出现内存溢出（OOM）错误，流水线利用离线预计算的统计数据（`stats.xls`）。在 PyTorch `DataLoader` 中进行即时（on-the-fly）缩放处理，确保内存效率及零数据泄露。
* **CNN-LSTM 混合架构**：采用 1D-CNN 层提取局部时空特征，随后接 LSTM 层以捕获长期的序列依赖关系。
* **序列验证机制**：在数据准备阶段实施严格的滑动窗口机制，确保在训练或验证阶段不会泄露未来信息。

## 📁 仓库结构

```text
Industrial-Signal-Forecaster/
├── data/                    # 数据集存储及数据架构示例
├── features/                # 特征工程产物（选定列、统计信息）
├── logs/                    # 训练及相关性分析日志
├── src/                     # 核心源代码（模型、训练与预测）
├── notebooks/               # 探索性数据分析 (EDA) 及实验记录
├── requirements.txt         # 环境依赖
└── README.md                # 项目文档
```
