import pandas as pd
import numpy as np
import torch
import os

from models import (NeuralNetwork, TimeSeriesTransformer, KalMan,
                    LSTMGRUForecaster, CNNLSTMForecaster)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'lstm'

    # 给定固定格式输入情况下，输出的情况
    input_data = pd.read_excel('train.csv').iloc[100:155, 1:]

    selected_123 = np.array(['信号42', '信号43', '信号73', '信号76', '信号123'], dtype=object)
    selected_124 = np.array(
        ['信号2', '信号6', '信号16', '信号22', '信号23', '信号24', '信号25', '信号26', '信号27', '信号28', '信号50',
         '信号53', '信号56', '信号60', '信号61', '信号63', '信号64', '信号77', '信号86', '信号96', '信号104', '信号106',
         '信号119', '信号124'], dtype=object)

    index_123 = [int(i[2:]) for i in selected_123]
    index_124 = [int(i[2:]) for i in selected_124]

    data_123 = input_data[selected_123]
    data_124 = input_data[selected_124]

    # 读取训练时生成的 stats.csv
    # 假设使用pandas索引读取机制
    stats = pd.read_excel('stats.csv', index_col=0)
    max_use_123 = stats[['max']].loc[index_123]
    min_use_123 = stats[['min']].loc[index_123]
    max_use_124 = stats[['max']].loc[index_124]
    min_use_124 = stats[['min']].loc[index_124]

    data_123_nor = (data_123 - min_use_123.values.T) / (max_use_123.values - min_use_123.values).T
    data_124_nor = (data_124 - min_use_124.values.T) / (max_use_124.values - min_use_124.values).T

    # ----------------------------------------------------
    # 第一部分：Jupyter前部的Demo预测部分 (KalMan Ensemble)
    # ----------------------------------------------------
    input_ml_123 = data_123_nor.iloc[-24:, :].values
    input_ml_124 = data_124_nor.iloc[-24:, :].values

    input_window_demo = 48
    input_ts_123_demo = data_123_nor.iloc[-1 * input_window_demo:, :].values.reshape(1, input_window_demo, -1)

    model = NeuralNetwork(5)
    output_123 = model(torch.tensor(input_ml_123, dtype=torch.float32)).squeeze().detach().numpy()

    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    pred_window = 24
    nhead = 4

    model_1 = TimeSeriesTransformer(5, hidden_dim, num_layers, output_dim, pred_window, nhead)
    output_123_ts1 = model_1(torch.tensor(input_ts_123_demo, dtype=torch.float32)).squeeze().detach().numpy()

    model_2 = TimeSeriesTransformer(5, hidden_dim, num_layers, output_dim, pred_window, nhead)
    output_123_ts2 = model_2(torch.tensor(input_ts_123_demo, dtype=torch.float32)).squeeze().detach().numpy()

    model_3 = TimeSeriesTransformer(5, hidden_dim, num_layers, output_dim, pred_window, nhead)
    output_123_ts3 = model_3(torch.tensor(input_ts_123_demo, dtype=torch.float32)).squeeze().detach().numpy()

    input_KALMAN = np.stack([output_123, output_123_ts1, output_123_ts2, output_123_ts3]).T
    output_KALMAN = KalMan(input_KALMAN)

    pre_signal_123 = output_KALMAN * (max_use_123.loc['信号123' if '信号123' in max_use_123.index else 123].values[0] -
                                      min_use_123.loc['信号123' if '信号123' in min_use_123.index else 123].values[0]) + \
                     min_use_123.loc['信号123' if '信号123' in min_use_123.index else 123].values[0]
    print("Demo预估(pre_signal_123): \n", pre_signal_123)

    # ----------------------------------------------------
    # 第二部分：调用最佳权重模型的预测推断流程
    # ----------------------------------------------------
    input_window_len = 96  # 默认 96 依据全局变量

    # 获取原始数据的全局归一化结果用于最终的预测
    data_normalized = pd.read_excel('train.csv').iloc[:, 1:]
    data_normalized = (data_normalized - stats['min']) / (stats['max'] - stats['min'])

    print("\n--- 读取保存的最佳模型权重进行推断 ---")

    # 预测信号 123
    model_123 = LSTMGRUForecaster(input_dim=5, lstm_hidden_dim=80, gru_hidden_dim=32, lstm_layers=3, gru_layers=3,
                                  output_dim=1, pred_window=24, dropout=0.15).to(device)
    model_path_123 = os.path.join(output_dir, 'best_lstm_gru_123.pth')
    if os.path.exists(model_path_123):
        model_123.load_state_dict(torch.load(model_path_123, map_location=device))
    model_123.eval()

    with torch.no_grad():
        input_ts_123 = data_normalized[selected_123].iloc[-input_window_len:, :].values.reshape(1, input_window_len, -1)
        input_ts_123_tensor = torch.tensor(input_ts_123, dtype=torch.float32).to(device)
        out_123 = model_123(input_ts_123_tensor).squeeze().cpu().numpy()
        out_123 = np.nan_to_num(out_123, nan=0.0, posinf=0.0, neginf=0.0)

        # 兼容取值方式
        key_123 = '信号123' if '信号123' in max_use_123.index else 123
        max_val = max_use_123.loc[key_123].values[0]
        min_val = min_use_123.loc[key_123].values[0]
        output_123_denorm = out_123 * (max_val - min_val) + min_val

        output_path_123 = os.path.join(output_dir, 'output_lstm_gru_123.pth')
        torch.save({'output': out_123, 'output_denorm': output_123_denorm}, output_path_123)
        print(f"预测信号123（去归一化）: {output_123_denorm}")

    # 预测信号 124
    model_124 = CNNLSTMForecaster(input_dim=24, lstm_hidden_dim=256, num_layers=3, output_dim=1, pred_window=24,
                                  dropout=0.2).to(device)
    model_path_124 = os.path.join(output_dir, 'best_cnn_lstm_124.pth')
    if os.path.exists(model_path_124):
        model_124.load_state_dict(torch.load(model_path_124, map_location=device))
    model_124.eval()

    with torch.no_grad():
        input_ts_124 = data_normalized[selected_124].iloc[-input_window_len:, :].values.reshape(1, input_window_len, -1)
        input_ts_124_tensor = torch.tensor(input_ts_124, dtype=torch.float32).to(device)
        out_124 = model_124(input_ts_124_tensor).squeeze().cpu().numpy()
        out_124 = np.nan_to_num(out_124, nan=0.0, posinf=0.0, neginf=0.0)

        key_124 = '信号124' if '信号124' in max_use_124.index else 124
        max_val = max_use_124.loc[key_124].values[0]
        min_val = min_use_124.loc[key_124].values[0]
        output_124_denorm = out_124 * (max_val - min_val) + min_val

        output_path_124 = os.path.join(output_dir, 'output_cnn_lstm_124.pth')
        torch.save({'output': out_124, 'output_denorm': output_124_denorm}, output_path_124)
        print(f"预测信号124（去归一化）: {output_124_denorm}")