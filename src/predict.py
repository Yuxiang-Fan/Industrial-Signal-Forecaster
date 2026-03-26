import os
import torch
import numpy as np
import pandas as pd
from models import LSTMGRUForecaster, CNNLSTMForecaster

def run_inference():
    """
    工业信号预测推断主程序。
    包含数据标准化、模型权重加载、前向传播推断以及结果逆归一化过程。
    """
    # 硬件设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'checkpoints'  # 统一存放模型权重的路径
    
    # 定义不同目标信号的特征子集（基于特征筛选阶段的产物）
    selected_features_123 = ['信号42', '信号43', '信号73', '信号76', '信号123']
    selected_features_124 = [
        '信号2', '信号6', '信号16', '信号22', '信号23', '信号24', '信号25', '信号26', 
        '信号27', '信号28', '信号50', '信号53', '信号56', '信号60', '信号61', '信号63', 
        '信号64', '信号77', '信号86', '信号96', '信号104', '信号106', '信号119', '信号124'
    ]

    # 加载训练阶段保存的统计特征，确保推断时的数据分布与训练集严格一致
    if not os.path.exists('stats.csv'):
        print("错误：找不到统计配置文件 stats.csv，无法进行归一化。")
        return
        
    stats = pd.read_excel('stats.csv', index_col=0)
    
    # 载入待预测原始数据
    input_df = pd.read_excel('train.csv').iloc[:, 1:]
    
    # ----------------------------------------------------
    # 核心环节一：信号 123 趋势预测 (基于混合循环网络)
    # ----------------------------------------------------
    print("\n[任务 1] 正在进行信号 123 的实时推断...")
    
    # 初始化 LSTM-GRU 模型架构
    model_123 = LSTMGRUForecaster(
        input_dim=len(selected_features_123), 
        lstm_hidden_dim=80, 
        gru_hidden_dim=32, 
        lstm_layers=3, 
        gru_layers=3,
        output_dim=1, 
        pred_window=24, 
        dropout=0.15
    ).to(device)

    # 载入最优模型权重
    weight_path_123 = os.path.join(output_dir, 'best_lstm_gru_123.pth')
    if os.path.exists(weight_path_123):
        model_123.load_state_dict(torch.load(weight_path_123, map_location=device))
        model_123.eval()
    else:
        print(f"警告：未找到预训练权重 {weight_path_123}")

    with torch.no_grad():
        # 数据预处理：特征提取与 Min-Max 归一化
        sub_data_123 = input_df[selected_features_123]
        feat_min_123 = stats.loc[selected_features_123, 'min'].values
        feat_max_123 = stats.loc[selected_features_123, 'max'].values
        
        # 执行滑动窗口截取，输入长度为 96 个时间步
        raw_input_123 = sub_data_123.iloc[-96:, :].values
        norm_input_123 = (raw_input_123 - feat_min_123) / (feat_max_123 - feat_min_123 + 1e-9)
        
        # 转换为张量并移动至计算设备
        tensor_input_123 = torch.tensor(norm_input_123, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 模型推断
        pred_norm_123 = model_123(tensor_input_123).squeeze().cpu().numpy()
        
        # 结果逆归一化：将模型输出还原为真实的工业量纲
        target_max_123 = stats.loc['信号123', 'max']
        target_min_123 = stats.loc['信号123', 'min']
        final_pred_123 = pred_norm_123 * (target_max_123 - target_min_123) + target_min_123
        
        print(f"信号 123 未来 24 时间步预测完成。首位预测值: {final_pred_123[0]:.4f}")

    # ----------------------------------------------------
    # 核心环节二：信号 124 趋势预测 (基于时空混合 CNN-LSTM)
    # ----------------------------------------------------
    print("\n[任务 2] 正在进行信号 124 的实时推断...")

    model_124 = CNNLSTMForecaster(
        input_dim=len(selected_features_124), 
        lstm_hidden_dim=256, 
        num_layers=3, 
        output_dim=1, 
        pred_window=24,
        dropout=0.2
    ).to(device)

    weight_path_124 = os.path.join(output_dir, 'best_cnn_lstm_124.pth')
    if os.path.exists(weight_path_124):
        model_124.load_state_dict(torch.load(weight_path_124, map_location=device))
        model_124.eval()

    with torch.no_grad():
        # 处理 24 维的高维传感器特征输入
        sub_data_124 = input_df[selected_features_124]
        feat_min_124 = stats.loc[selected_features_124, 'min'].values
        feat_max_124 = stats.loc[selected_features_124, 'max'].values
        
        raw_input_124 = sub_data_124.iloc[-96:, :].values
        norm_input_124 = (raw_input_124 - feat_min_124) / (feat_max_124 - feat_min_124 + 1e-9)
        
        tensor_input_124 = torch.tensor(norm_input_124, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 针对异常值进行预处理防护，确保计算稳定性
        pred_norm_124 = model_124(tensor_input_124).squeeze().cpu().numpy()
        pred_norm_124 = np.nan_to_num(pred_norm_124, nan=0.0)
        
        target_max_124 = stats.loc['信号124', 'max']
        target_min_124 = stats.loc['信号124', 'min']
        final_pred_124 = pred_norm_124 * (target_max_124 - target_min_124) + target_min_124
        
        print(f"信号 124 未来 24 时间步预测完成。首位预测值: {final_pred_124[0]:.4f}")

    # ----------------------------------------------------
    # 结果持久化：保存预测产物用于可视化或下游控制逻辑
    # ----------------------------------------------------
    results = {
        'signal_123': final_pred_123,
        'signal_124': final_pred_124
    }
    torch.save(results, os.path.join(output_dir, 'latest_inference_results.pth'))
    print("\n所有推断结果已保存至本地归档。")

if __name__ == '__main__':
    run_inference()
