import pandas as pd
import numpy as np
import random
import os
import gc
import json
import copy
import sys
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, TensorDataset

from models import (TimeSeriesDataset, TimeSeriesDatasetWithNoise,
                    LSTMGRUForecaster, CNNLSTMForecaster, input_window)

# 设置所有随机种子（确保实验可重复）
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def split_ml_data(X, y, test_size=0.2, batch_size=1024):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))
    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def split_time_series(dataset, train_ratio=0.8, val_ratio=0.2, batch_size=64):
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
    def __init__(self, optimizer, warmup_epochs, base_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        super(WarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_epochs for _ in self.optimizer.param_groups]
        return [self.base_lr for _ in self.optimizer.param_groups]


def train_ts_model(model, train_loader, val_loader, device, model_name, epochs=150, patience=80):
    model = model.to(device)
    criterion = nn.HuberLoss(delta=0.5)
    base_lr = 0.00008 if '123' in model_name else 0.00007
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=3e-3)
    warmup_scheduler = WarmupLR(optimizer, warmup_epochs=15, base_lr=base_lr)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                             min_lr=1e-6)
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
            print(f'保存最佳指标: {metrics_path}')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'在轮次 {epoch + 1} 提前停止')
                break

        if epoch < 15:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_mse)

    loss_df = pd.DataFrame(loss_history)
    loss_df.to_csv(os.path.join(output_dir, 'loss_history_' + model_name + '.csv'), index=False)
    print(f'保存损失历史: {os.path.join(output_dir, "loss_history_" + model_name + ".csv")}')

    model.load_state_dict(best_model_state)
    return model, best_metrics


if __name__ == '__main__':
    # 设置输出文件夹
    output_dir = 'lstm'
    os.makedirs(output_dir, exist_ok=True)

    # 环境配置和显存清理
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    print("CUDA可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU名称:", torch.cuda.get_device_name(0))
        print("GPU总内存:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
        torch.cuda.empty_cache()
        gc.collect()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备:", device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.8)
    torch.cuda.empty_cache()
    gc.collect()

    # 1、数据读取
    data = pd.read_excel('train.csv').iloc[:, 1:]

    # 2、统计量
    stats = data.agg(['max', 'min']).T
    max1 = stats['max']
    min1 = stats['min']
    stats.to_csv('stats.csv')

    # 3、向量化归一化
    data_normalized = (data - min1) / (max1 - min1)
    data_normalized = data_normalized.dropna(axis=1, how='all')

    data_normalized['target_123_ml'] = data_normalized['信号123'].shift(-24)
    data_normalized['target_124_ml'] = data_normalized['信号124'].shift(-24)

    # 4、选用的特征如下
    selected_123 = np.array(['信号42', '信号43', '信号73', '信号76', '信号123'], dtype=object)
    selected_124 = np.array(
        ['信号2', '信号6', '信号16', '信号22', '信号23', '信号24', '信号25', '信号26', '信号27', '信号28', '信号50',
         '信号53', '信号56', '信号60', '信号61', '信号63', '信号64', '信号77', '信号86', '信号96', '信号104', '信号106',
         '信号119', '信号124'], dtype=object)

    # 6 机器学习使用的数据集
    X_123 = data_normalized[selected_123].values.astype(np.float32)
    y_123 = data_normalized['target_123_ml'].values.astype(np.float32)

    X_124 = data_normalized[selected_124].values.astype(np.float32)
    y_124 = data_normalized['target_124_ml'].values.astype(np.float32)

    train_loader_123, test_loader_123 = split_ml_data(X_123, y_123)
    train_loader_124, test_loader_124 = split_ml_data(X_124, y_124)

    # 7、时间序列模型使用的数据集
    input_window_len = input_window if 'input_window' in globals() else 96
    output_window_len = 24

    dataset_123_base = TimeSeriesDataset(data_normalized[selected_123], data_normalized[['信号123']],
                                         input_window=input_window_len, pred_window=output_window_len, stride=24)
    dataset_124_base = TimeSeriesDataset(data_normalized[selected_124], data_normalized[['信号124']],
                                         input_window=input_window_len, pred_window=output_window_len, stride=24)

    dataset_123 = TimeSeriesDatasetWithNoise(dataset_123_base, noise_std=0.0)
    dataset_124 = TimeSeriesDatasetWithNoise(dataset_124_base, noise_std=0.01)

    train_loader_ts_123, val_loader_ts_123, test_loader_ts_123 = split_time_series(dataset_123)
    train_loader_ts_124, val_loader_ts_124, test_loader_ts_124 = split_time_series(dataset_124)

    # 训练信号 123 模型（LSTM+GRU）
    print("为信号123训练LSTM+GRU模型...")
    input_dim_123 = len(selected_123)
    lstm_hidden_dim_123 = 80
    gru_hidden_dim_123 = 32
    lstm_layers = 3
    gru_layers = 3
    output_dim = 1
    pred_window = output_window_len

    try:
        model_123 = LSTMGRUForecaster(input_dim=input_dim_123, lstm_hidden_dim=lstm_hidden_dim_123,
                                      gru_hidden_dim=gru_hidden_dim_123,
                                      lstm_layers=lstm_layers, gru_layers=gru_layers, output_dim=output_dim,
                                      pred_window=pred_window, dropout=0.15).to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            device = torch.device('cpu')
            model_123 = LSTMGRUForecaster(input_dim=input_dim_123, lstm_hidden_dim=lstm_hidden_dim_123,
                                          gru_hidden_dim=gru_hidden_dim_123,
                                          lstm_layers=lstm_layers, gru_layers=gru_layers, output_dim=output_dim,
                                          pred_window=pred_window, dropout=0.15).to(device)
            dataset_123.dataset.X = torch.tensor(dataset_123.dataset.X, dtype=torch.float32).cpu()
            train_loader_ts_123 = DataLoader(dataset_123, batch_size=64, shuffle=True)
            val_loader_ts_123 = DataLoader(dataset_123, batch_size=64, shuffle=False)
        else:
            raise e

    model_123, best_metrics_123 = train_ts_model(model_123, train_loader_ts_123, val_loader_ts_123, device,
                                                 'lstm_gru_123')

    del model_123
    gc.collect()
    torch.cuda.empty_cache()

    # 训练信号 124 模型（CNN+LSTM）
    print("为信号124训练CNN+LSTM模型...")
    input_dim_124 = len(selected_124)
    lstm_hidden_dim_124 = 256
    num_layers = 3

    try:
        model_124 = CNNLSTMForecaster(input_dim=input_dim_124, lstm_hidden_dim=lstm_hidden_dim_124,
                                      num_layers=num_layers,
                                      output_dim=output_dim, pred_window=pred_window, dropout=0.2).to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            device = torch.device('cpu')
            model_124 = CNNLSTMForecaster(input_dim=input_dim_124, lstm_hidden_dim=lstm_hidden_dim_124,
                                          num_layers=num_layers,
                                          output_dim=output_dim, pred_window=pred_window, dropout=0.2).to(device)
            dataset_124.dataset.X = torch.tensor(dataset_124.dataset.X, dtype=torch.float32).cpu()
            train_loader_ts_124 = DataLoader(dataset_124, batch_size=64, shuffle=True)
            val_loader_ts_124 = DataLoader(dataset_124, batch_size=64, shuffle=False)
        else:
            raise e

    model_124, best_metrics_124 = train_ts_model(model_124, train_loader_ts_124, val_loader_ts_124, device,
                                                 'cnn_lstm_124')

    del model_124
    gc.collect()
    torch.cuda.empty_cache()

    # 完成提示
    print("训练完成！")
    print("信号123（LSTM+GRU）的最佳指标:", best_metrics_123)
    print("信号124（CNN+LSTM）的最佳指标:", best_metrics_124)