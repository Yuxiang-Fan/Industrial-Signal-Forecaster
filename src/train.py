import pandas as pd
import numpy as np
import random
import os
import gc
import json
import copy
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, TensorDataset

# 从 models.py 导入数据集类、模型类和全局超参数
from models import (TimeSeriesDataset, TimeSeriesDatasetWithNoise,
                    LSTMGRUForecaster, CNNLSTMForecaster, input_window)

# ==========================================
# 1. 全局随机种子设置
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ==========================================
# 2. 数据处理与调度器辅助函数
# ==========================================
def split_ml_data(X, y, test_size=0.2, batch_size=1024):
    """用于基础机器学习模型的随机打乱切分 (非时序滚动)"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def split_time_series(dataset, batch_size=64):
    """将时序数据集按 8:1:1 划分为训练集、验证集和测试集"""
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


class WarmupLR(optim.lr_scheduler._LRScheduler):
    """带 Warmup 预热的学习率调度器"""
    def __init__(self, optimizer, warmup_epochs, base_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        super(WarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_epochs for _ in self.optimizer.param_groups]
        return [self.base_lr for _ in self.optimizer.param_groups]


# ==========================================
# 3. 核心训练函数
# ==========================================
def train_ts_model(model, train_loader, val_loader, device, model_name, output_dir, epochs=150, patience=80):
    model = model.to(device)
    criterion = nn.HuberLoss(delta=0.5)
    # 根据模型分配不同的基础学习率
    base_lr = 0.00008 if '123' in model_name else 0.00007
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=3e-3)
    
    warmup_scheduler = WarmupLR(optimizer, warmup_epochs=15, base_lr=base_lr)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    scaler = amp.GradScaler()

    best_mse = float('inf')
    counter = 0
    best_model_state = None
    best_metrics = {}
    loss_history = {'epoch': [], 'train_loss': [], 'val_mse': []}

    for epoch in range(epochs):
        model.train()
        train_loader.dataset.training = True
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with amp.autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)

        model.eval()
        train_loader.dataset.training = False
        total_sse = 0.0
        total_num = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                outputs = model(X_batch)
                sse = F.mse_loss(outputs, y_batch, reduction='sum').item()
                num_elements = outputs.numel()
                total_sse += sse
                total_num += num_elements
                
        val_mse = total_sse / total_num if total_num > 0 else float('inf')

        loss_history['epoch'].append(epoch + 1)
        loss_history['train_loss'].append(train_loss)
        loss_history['val_mse'].append(val_mse)
        print(f'轮次 {epoch + 1}/{epochs}, 训练损失: {train_loss:.6f}, 验证MSE: {val_mse:.6f}')

        # 验证指标提升，保存最佳模型
        if val_mse < best_mse:
            best_mse = val_mse
            best_model_state = copy.deepcopy(model.state_dict())
            best_metrics = {'val_mse': val_mse, 'epoch': epoch + 1}
            
            model_path = os.path.join(output_dir, 'best_' + model_name + '.pth')
            metrics_path = os.path.join(output_dir, 'best_metrics_' + model_name + '.json')
            
            torch.save(best_model_state, model_path)
            with open(metrics_path, 'w') as f:
                json.dump(best_metrics, f, indent=4)
                
            print(f'保存最佳模型: {model_path} (验证MSE: {best_mse:.6f})')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'在轮次 {epoch + 1} 触发早停机制 (Early Stopping)')
                break

        # 更新学习率
        if epoch < 15:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_mse)

    # 保存训练过程的 Loss 曲线数据
    loss_df = pd.DataFrame(loss_history)
    loss_csv_path = os.path.join(output_dir, 'loss_history_' + model_name + '.csv')
    loss_df.to_csv(loss_csv_path, index=False)
    
    model.load_state_dict(best_model_state)
    return model, best_metrics


# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == '__main__':
    # ==========================================
    # 占位符配置区：请在此处替换为真实的 .xls 文件路径
    # ==========================================
    TRAIN_DATA_PATH = 'path/to/your/train_data.xls'  # 训练集输入数据
    STATS_DATA_PATH = 'path/to/your/stats.xls'       # 用于输出的统计最值数据
    
    output_dir = 'lstm'
    os.makedirs(output_dir, exist_ok=True)

    # 环境配置和显存清理
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- 启动训练流程，使用设备: {device} ---")
    
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.8)
    torch.cuda.empty_cache()
    gc.collect()

    # 1. 数据读取 (.xls) (假设第一列是序号或时间戳，将其过滤，只取后续特征)
    print("正在加载训练集数据...")
    data = pd.read_excel(TRAIN_DATA_PATH).iloc[:, 1:]

    # 2. 计算并保存统计量 (.xls)
    stats = data.agg(['max', 'min']).T
    stats.to_excel(STATS_DATA_PATH)
    print(f"已计算全局极值并保存至 {STATS_DATA_PATH}")

    # 3. 向量化归一化处理
    max1 = stats['max']
    min1 = stats['min']
    data_normalized = (data - min1) / (max1 - min1)
    data_normalized = data_normalized.dropna(axis=1, how='all')

    # 为非时序模型构造平移标签 (超前预测24步)
    data_normalized['target_123_ml'] = data_normalized['信号123'].shift(-24)
    data_normalized['target_124_ml'] = data_normalized['信号124'].shift(-24)

    # 4. 选用的特征集列名
    selected_123 = np.array(['信号42', '信号43', '信号73', '信号76', '信号123'], dtype=object)
    selected_124 = np.array([
        '信号2', '信号6', '信号16', '信号22', '信号23', '信号24', '信号25', '信号26', '信号27', '信号28', '信号50',
        '信号53', '信号56', '信号60', '信号61', '信号63', '信号64', '信号77', '信号86', '信号96', '信号104', '信号106',
        '信号119', '信号124'
    ], dtype=object)

    # -------------------------------------------------------------------
    # [保留备用] 机器学习使用的数据集 (本脚本主体执行时序预测，这部分作保留以供对比)
    # X_123 = data_normalized[selected_123].values.astype(np.float32)
    # y_123 = data_normalized['target_123_ml'].values.astype(np.float32)
    # train_loader_123_ml, test_loader_123_ml = split_ml_data(X_123, y_123)
    # -------------------------------------------------------------------

    # 5. 构建时间序列模型使用的数据集
    print("正在构建时序滑动窗口数据集...")
    input_window_len = input_window 
    output_window_len = 24

    dataset_123_base = TimeSeriesDataset(
        data_normalized[selected_123], data_normalized[['信号123']],
        input_window=input_window_len, pred_window=output_window_len, stride=24
    )
    dataset_124_base = TimeSeriesDataset(
        data_normalized[selected_124], data_normalized[['信号124']],
        input_window=input_window_len, pred_window=output_window_len, stride=24
    )

    # 信号124 加入适量数据增强高斯噪声
    dataset_123 = TimeSeriesDatasetWithNoise(dataset_123_base, noise_std=0.0)
    dataset_124 = TimeSeriesDatasetWithNoise(dataset_124_base, noise_std=0.01)

    train_loader_ts_123, val_loader_ts_123, test_loader_ts_123 = split_time_series(dataset_123)
    train_loader_ts_124, val_loader_ts_124, test_loader_ts_124 = split_time_series(dataset_124)

    # ==========================================
    # 模型训练 A: 信号 123 模型 (LSTM+GRU)
    # ==========================================
    print("\n>>> 开始为 [信号123] 训练 LSTM+GRU 模型...")
    input_dim_123 = len(selected_123)
    lstm_hidden_dim_123 = 80
    gru_hidden_dim_123 = 32
    
    try:
        model_123 = LSTMGRUForecaster(
            input_dim=input_dim_123, lstm_hidden_dim=lstm_hidden_dim_123, gru_hidden_dim=gru_hidden_dim_123,
            lstm_layers=3, gru_layers=3, output_dim=1, pred_window=output_window_len, dropout=0.15
        ).to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("CUDA 显存不足，回退至 CPU 进行模型初始化...")
            device = torch.device('cpu')
            model_123 = LSTMGRUForecaster(
                input_dim=input_dim_123, lstm_hidden_dim=lstm_hidden_dim_123, gru_hidden_dim=gru_hidden_dim_123,
                lstm_layers=3, gru_layers=3, output_dim=1, pred_window=output_window_len, dropout=0.15
            ).to(device)
        else:
            raise e

    model_123, best_metrics_123 = train_ts_model(
        model_123, train_loader_ts_123, val_loader_ts_123, device, 'lstm_gru_123', output_dir
    )

    del model_123
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()

    # ==========================================
    # 模型训练 B: 信号 124 模型 (CNN+LSTM)
    # ==========================================
    print("\n>>> 开始为 [信号124] 训练 CNN+LSTM 模型...")
    input_dim_124 = len(selected_124)
    lstm_hidden_dim_124 = 256
    
    try:
        model_124 = CNNLSTMForecaster(
            input_dim=input_dim_124, lstm_hidden_dim=lstm_hidden_dim_124, num_layers=3,
            output_dim=1, pred_window=output_window_len, dropout=0.2
        ).to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("CUDA 显存不足，回退至 CPU 进行模型初始化...")
            device = torch.device('cpu')
            model_124 = CNNLSTMForecaster(
                input_dim=input_dim_124, lstm_hidden_dim=lstm_hidden_dim_124, num_layers=3,
                output_dim=1, pred_window=output_window_len, dropout=0.2
            ).to(device)
        else:
            raise e

    model_124, best_metrics_124 = train_ts_model(
        model_124, train_loader_ts_124, val_loader_ts_124, device, 'cnn_lstm_124', output_dir
    )

    del model_124
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()

    # ==========================================
    # 训练结束总结
    # ==========================================
    print("\n--- 所有训练任务已完成！ ---")
    print(f"信号123 (LSTM+GRU) 的最佳验证指标: {best_metrics_123}")
    print(f"信号124 (CNN+LSTM) 的最佳验证指标: {best_metrics_124}")
