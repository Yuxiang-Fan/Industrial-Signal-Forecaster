# Dataset Information

Due to data licensing and GitHub file size constraints, the full training and testing datasets are not hosted in this repository. 

Instead, a lightweight `sample_data.xls` is provided in this directory solely to demonstrate the data structure and schema required for the CNN-LSTM model.

## 🔗 Original Data Source

The complete dataset originates from the **2025 "中控杯" 工业AI创新挑战赛** (2025 Supcon Cup Industrial AI Innovation Challenge).
* **Track:** 工业时序预测模型创新及应用赛道 (Industrial Time-Series Forecasting Model Innovation and Application)

If you wish to run this pipeline on the full dataset, please search for the competition name online to locate the official download portal. Once downloaded, place your `.xls` files into this `data/` directory and ensure the file paths in `train.py` and `predict.py` point to them.