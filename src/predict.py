import pandas as pd
import numpy as np
import torch
import os

from models import LSTMGRUForecaster, CNNLSTMForecaster, input_window

if __name__ == '__main__':
    # 1. 基础设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'lstm'  # 模型权重存放目录
    
    print(f"--- 开始预测流程，使用设备: {device} ---")

    # ==========================================
    # 占位符配置区：请在此处替换为真实的 .xls 文件路径
    # ==========================================
    TEST_DATA_PATH = 'path/to/your/test_data.xls'  # 测试集输入数据
    STATS_DATA_PATH = 'path/to/your/stats.xls'     # 训练时保存的统计最值数据
    
    # 定义特征列 (保持与训练时对齐)
    selected_123 = np.array(['信号42', '信号43', '信号73', '信号76', '信号123'], dtype=object)
    selected_124 = np.array([
        '信号2', '信号6', '信号16', '信号22', '信号23', '信号24', '信号25', '信号26', '信号27', '信号28', '信号50',
        '信号53', '信号56', '信号60', '信号61', '信号63', '信号64', '信号77', '信号86', '信号96', '信号104', '信号106',
        '信号119', '信号124'
    ], dtype=object)

    # 提取整型的特征索引，用于从 stats 中提取对应的最值统计量
    index_123 = [int(i[2:]) for i in selected_123]
    index_124 = [int(i[2:]) for i in selected_124]

    # 2. 读取外部表格数据 (.xls格式)
    print("正在读取外部原始数据和统计信息...")
    
    # 读取输入数据 (如果您真实的xls表格中包含时间戳等非特征列，请通过 iloc 或 drop 进行相应处理)
    input_data = pd.read_excel(TEST_DATA_PATH)
    
    # 读取统计量数据 (假设索引被保存在了第一列，所以使用 index_col=0)
    stats = pd.read_excel(STATS_DATA_PATH, index_col=0)

    # 提取统计最值用于反归一化
    max_use_123 = stats[['max']].loc[index_123]
    min_use_123 = stats[['min']].loc[index_123]
    max_use_124 = stats[['max']].loc[index_124]
    min_use_124 = stats[['min']].loc[index_124]

    # 利用全局统计量进行归一化
    data_normalized = (input_data - stats['min'].values) / (stats['max'].values - stats['min'].values)

    print("\n--- 加载最佳模型权重进行推断 ---")

    # ==========================================
    # 预测信号 123 (LSTM + GRU)
    # ==========================================
    model_123 = LSTMGRUForecaster(
        input_dim=len(selected_123), lstm_hidden_dim=80, gru_hidden_dim=32, 
        lstm_layers=3, gru_layers=3, output_dim=1, pred_window=24, dropout=0.15
    ).to(device)
    
    model_path_123 = os.path.join(output_dir, 'best_lstm_gru_123.pth')
    if os.path.exists(model_path_123):
        model_123.load_state_dict(torch.load(model_path_123, map_location=device))
        print(f"成功加载模型: {model_path_123}")
    else:
        print(f"警告: 未找到模型文件 {model_path_123}")
        
    model_123.eval()

    with torch.no_grad():
        # 截取最后 input_window 个时间步的数据作为输入
        input_ts_123 = data_normalized[selected_123].iloc[-input_window:, :].values.reshape(1, input_window, -1)
        input_ts_123_tensor = torch.tensor(input_ts_123, dtype=torch.float32).to(device)
        
        out_123 = model_123(input_ts_123_tensor).squeeze().cpu().numpy()
        out_123 = np.nan_to_num(out_123, nan=0.0, posinf=0.0, neginf=0.0)

        # 反归一化
        max_val = max_use_123.loc[123].values[0]
        min_val = min_use_123.loc[123].values[0]
        output_123_denorm = out_123 * (max_val - min_val) + min_val

        output_file_123 = os.path.join(output_dir, 'output_lstm_gru_123.pth')
        torch.save({'output': out_123, 'output_denorm': output_123_denorm}, output_file_123)
        print(f"-> 预测信号123（去归一化结果）保存至 {output_file_123}")

    # ==========================================
    # 预测信号 124 (CNN + LSTM)
    # ==========================================
    model_124 = CNNLSTMForecaster(
        input_dim=len(selected_124), lstm_hidden_dim=256, 
        num_layers=3, output_dim=1, pred_window=24, dropout=0.2
    ).to(device)
    
    model_path_124 = os.path.join(output_dir, 'best_cnn_lstm_124.pth')
    if os.path.exists(model_path_124):
        model_124.load_state_dict(torch.load(model_path_124, map_location=device))
        print(f"成功加载模型: {model_path_124}")
    else:
        print(f"警告: 未找到模型文件 {model_path_124}")
        
    model_124.eval()

    with torch.no_grad():
        # 截取最后 input_window 个时间步的数据作为输入
        input_ts_124 = data_normalized[selected_124].iloc[-input_window:, :].values.reshape(1, input_window, -1)
        input_ts_124_tensor = torch.tensor(input_ts_124, dtype=torch.float32).to(device)
        
        out_124 = model_124(input_ts_124_tensor).squeeze().cpu().numpy()
        out_124 = np.nan_to_num(out_124, nan=0.0, posinf=0.0, neginf=0.0)

        # 反归一化
        max_val = max_use_124.loc[124].values[0]
        min_val = min_use_124.loc[124].values[0]
        output_124_denorm = out_124 * (max_val - min_val) + min_val

        output_file_124 = os.path.join(output_dir, 'output_cnn_lstm_124.pth')
        torch.save({'output': out_124, 'output_denorm': output_124_denorm}, output_file_124)
        print(f"-> 预测信号124（去归一化结果）保存至 {output_file_124}")
        
    print("\n--- 推断流程结束 ---")
