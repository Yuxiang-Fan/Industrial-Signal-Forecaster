import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# 全局默认输入窗口大小
input_window = 96

class TimeSeriesDataset(Dataset):
    """
    基础时间序列数据集
    """
    def __init__(self, X, y, input_window=input_window, pred_window=24, stride=1):
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y = y.values if isinstance(y, pd.DataFrame) else y
        self.input_window = input_window
        self.pred_window = pred_window
        self.stride = stride
        # 预先计算所有有效采样的起始索引，支持通过步长(stride)进行滑动采样
        self.indices = list(range(0, len(self.X) - input_window - pred_window + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        X = self.X[i : i + self.input_window]
        y = self.y[i + self.input_window : i + self.input_window + self.pred_window]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TimeSeriesDatasetWithNoise(Dataset):
    """
    带数据增强（高斯噪声）的时间序列数据集包装器
    """
    def __init__(self, dataset, noise_std=0.01):
        self.dataset = dataset
        self.noise_std = noise_std
        self.training = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.training and self.noise_std > 0:
            noise = torch.normal(0, self.noise_std, size=x.shape)
            x = x + noise
        return x, y


class NeuralNetwork(nn.Module):
    """
    机器学习模型的标准接口形式 (多层感知机)
    """
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


class TimeSeriesTransformer(nn.Module):
    """
    基于Transformer的时间序列预测模型
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pred_window, nhead=4):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim * pred_window)

    def forward(self, x):
        # [batch_size, seq_len, input_dim] -> [batch_size, seq_len, hidden_dim]
        x = self.input_projection(x)  
        # [batch_size, seq_len, hidden_dim]
        out = self.transformer(x)  
        # 取最后一个时间步的输出特征进行预测
        out = self.fc(out[:, -1, :])  
        # [batch_size, pred_window, output_dim]
        return out.view(x.size(0), -1, 1)  


class LSTMGRUForecaster(nn.Module):
    """
    串联 LSTM 和 GRU 的混合递归神经网络，带残差连接
    (用于处理 信号123)
    """
    def __init__(self, input_dim, lstm_hidden_dim, gru_hidden_dim, lstm_layers, gru_layers, output_dim, pred_window, dropout=0.15):
        super(LSTMGRUForecaster, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.gru = nn.GRU(
            input_size=lstm_hidden_dim, hidden_size=gru_hidden_dim, num_layers=gru_layers,
            batch_first=True, dropout=dropout if gru_layers > 1 else 0.0
        )
        self.residual_fc = nn.Linear(input_dim, gru_hidden_dim)
        self.fc = nn.Linear(gru_hidden_dim, pred_window * output_dim)
        self.pred_window = pred_window
        self.output_dim = output_dim

    def forward(self, x):
        # 提取最后一个时间步的输入作残差连接
        residual = self.residual_fc(x[:, -1, :])
        
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(lstm_out)
        
        # 将GRU最后的隐藏状态与输入残差相加
        last_hidden = gru_out[:, -1, :] + residual
        
        out = self.fc(last_hidden)
        out = out.view(x.size(0), self.pred_window, self.output_dim)
        return out


class CNNLSTMForecaster(nn.Module):
    """
    一维卷积 (CNN) 提取局部特征后接入 LSTM 的混合模型，带残差连接
    (用于处理 信号124)
    """
    def __init__(self, input_dim, lstm_hidden_dim, num_layers, output_dim, pred_window, dropout=0.2):
        super(CNNLSTMForecaster, self).__init__()
        # 1D Convolution 块
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=lstm_hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.residual_fc = nn.Linear(input_dim, lstm_hidden_dim)
        self.fc = nn.Linear(lstm_hidden_dim, pred_window * output_dim)
        self.pred_window = pred_window
        self.output_dim = output_dim

    def forward(self, x):
        # 提取最后一个时间步的输入作残差连接
        residual = self.residual_fc(x[:, -1, :])
        
        # Conv1d 要求输入 shape 为 (batch, channel, seq_len)
        x = x.transpose(1, 2)  
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # 还原回 (batch, seq_len, channel) 以输入 LSTM
        x = x.transpose(1, 2)  
        
        lstm_out, (h_n, _) = self.lstm(x)
        
        # 取 LSTM 最后一层的最后一个隐藏状态并加上残差
        last_hidden = h_n[-1] + residual
        
        out = self.fc(last_hidden)
        out = out.view(x.size(0), self.pred_window, self.output_dim)
        return out
