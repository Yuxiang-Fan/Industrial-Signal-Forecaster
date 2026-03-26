import os
import gc
import json
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Subset

from models import (TimeSeriesDataset, TimeSeriesDatasetWithNoise,
                    LSTMGRUForecaster, CNNLSTMForecaster)

# 强制固定全局随机种子，保障实验结果具备完全的可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def split_time_series_chronologically(dataset, train_ratio=0.8, val_ratio=0.1, batch_size=64):
    """
    严格基于时间轴先后顺序划分数据集，严禁使用随机切分，杜绝未来信息泄露。
    """
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    # 按时间序列先后生成索引
    indices = list(range(total_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # 训练集内部打乱有助于梯度下降的稳定性，验证集与测试集严格保持时序
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

class WarmupLR(optim.lr_scheduler._LRScheduler):
    """
    自定义学习率预热调度器。
    在训练初期使用较小的学习率缓慢上升，防止初始化权重时梯度爆炸破坏模型结构。
    """
    def __init__(self, optimizer, warmup_epochs, base_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        super(WarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_epochs for _ in self.optimizer.param_groups]
        return [self.base_lr for _ in self.optimizer.param_groups]

def train_ts_model(model, train_loader, val_loader, device, model_name, output_dir, epochs=150, patience=80):
    """
    深度时间序列模型通用训练流水线。
    集成了自动混合精度加速、梯度裁剪、早停机制与动态学习率调整。
    """
    model = model.to(device)
    # 选用 Huber Loss 代替 MSE，增强对工业传感器异常尖峰噪声的鲁棒性
    criterion = nn.HuberLoss(delta=0.5)
    
    base_lr = 0.00008 if '123' in model_name else 0.00007
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=3e-3)
    
    warmup_scheduler = WarmupLR(optimizer, warmup_epochs=15, base_lr=base_lr)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # 初始化自动混合精度梯度缩放器，显著降低显存占用并加速训练
    scaler = amp.GradScaler()

    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    best_metrics = {}
    loss_history = {'epoch': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loader.dataset.dataset.training = True 
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 开启自动混合精度上下文
            with amp.autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # 全局梯度裁剪，防止深层循环网络发生梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)

        model.eval()
        train_loader.dataset.dataset.training = False 
        total_sse = 0.0
        total_num = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                outputs = model(X_batch)
                
                sse = F.mse_loss(outputs, y_batch, reduction='sum').item()
                total_sse += sse
                total_num += outputs.numel()
                
        val_mse = total_sse / total_num if total_num > 0 else float('inf')

        loss_history['epoch'].append(epoch + 1)
        loss_history['train_loss'].append(train_loss)
        loss_history['val_loss'].append(val_mse)
        print(f'迭代轮次 {epoch + 1:03d}/{epochs} | 训练集 Huber 损失: {train_loss:.6f} | 验证集 MSE 损失: {val_mse:.6f}')

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_model_state = copy.deepcopy(model.state_dict())
            best_metrics = {'val_mse': val_mse, 'epoch': epoch + 1}
            
            model_path = os.path.join(output_dir, f'best_{model_name}.pth')
            metrics_path = os.path.join(output_dir, f'best_metrics_{model_name}.json')
            
            torch.save(best_model_state, model_path)
            with open(metrics_path, 'w') as f:
                json.dump(best_metrics, f, indent=4)
                
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'连续 {patience} 轮验证集指标未提升，触发早停机制。')
                break

        if epoch < 15:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_mse)

    loss_df = pd.DataFrame(loss_history)
    loss_df.to_csv(os.path.join(output_dir, f'loss_history_{model_name}.csv'), index=False)

    model.load_state_dict(best_model_state)
    return model, best_metrics

if __name__ == '__main__':
    output_dir = 'checkpoints'
    os.makedirs(output_dir, exist_ok=True)

    # 优化 PyTorch 显存分配策略，缓解内存碎片化问题
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前计算设备: {device}")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    print("正在加载并初始化全局数据集...")
    data = pd.read_excel('train.csv').iloc[:, 1:]

    # 提取并持久化极值统计量，供预测推理阶段保持一致性缩放
    stats = data.agg(['max', 'min']).T
    stats.to_csv('stats.csv')

    data_normalized = (data - stats['min']) / (stats['max'] - stats['min'] + 1e-9)
    data_normalized = data_normalized.dropna(axis=1, how='all')

    selected_123 = ['信号42', '信号43', '信号73', '信号76', '信号123']
    selected_124 = [
        '信号2', '信号6', '信号16', '信号22', '信号23', '信号24', '信号25', '信号26', 
        '信号27', '信号28', '信号50', '信号53', '信号56', '信号60', '信号61', '信号63', 
        '信号64', '信号77', '信号86', '信号96', '信号104', '信号106', '信号119', '信号124'
    ]

    input_window_len = 96
    output_window_len = 24

    dataset_123_base = TimeSeriesDataset(
        data_normalized[selected_123], data_normalized[['信号123']],
        input_window=input_window_len, pred_window=output_window_len, stride=24
    )
    dataset_124_base = TimeSeriesDataset(
        data_normalized[selected_124], data_normalized[['信号124']],
        input_window=input_window_len, pred_window=output_window_len, stride=24
    )

    dataset_123 = TimeSeriesDatasetWithNoise(dataset_123_base, noise_std=0.0)
    dataset_124 = TimeSeriesDatasetWithNoise(dataset_124_base, noise_std=0.01)

    train_loader_ts_123, val_loader_ts_123, test_loader_ts_123 = split_time_series_chronologically(dataset_123)
    train_loader_ts_124, val_loader_ts_124, test_loader_ts_124 = split_time_series_chronologically(dataset_124)

    print("\n========== 启动任务一：信号 123 趋势建模 (LSTM-GRU) ==========")
    try:
        model_123 = LSTMGRUForecaster(
            input_dim=len(selected_123), lstm_hidden_dim=80, gru_hidden_dim=32,
            lstm_layers=3, gru_layers=3, output_dim=1, pred_window=output_window_len, dropout=0.15
        ).to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("警告：显存溢出，主动降级回退至 CPU 计算节点进行训练...")
            device = torch.device('cpu')
            model_123 = LSTMGRUForecaster(
                input_dim=len(selected_123), lstm_hidden_dim=80, gru_hidden_dim=32,
                lstm_layers=3, gru_layers=3, output_dim=1, pred_window=output_window_len, dropout=0.15
            ).to(device)
        else:
            raise e

    model_123, best_metrics_123 = train_ts_model(
        model_123, train_loader_ts_123, val_loader_ts_123, device, 'lstm_gru_123', output_dir
    )

    # 释放显存避免多任务并发导致崩溃
    del model_123
    gc.collect()
    torch.cuda.empty_cache()

    print("\n========== 启动任务二：高维信号 124 趋势建模 (CNN-LSTM) ==========")
    model_124 = CNNLSTMForecaster(
        input_dim=len(selected_124), lstm_hidden_dim=256, num_layers=3,
        output_dim=1, pred_window=output_window_len, dropout=0.2
    ).to(device)

    model_124, best_metrics_124 = train_ts_model(
        model_124, train_loader_ts_124, val_loader_ts_124, device, 'cnn_lstm_124', output_dir
    )

    print("\n========== 全局训练流水线执行完毕 ==========")
    print(f"信号 123 (LSTM-GRU) 最优验证集指标: {best_metrics_123}")
    print(f"信号 124 (CNN-LSTM) 最优验证集指标: {best_metrics_124}")
