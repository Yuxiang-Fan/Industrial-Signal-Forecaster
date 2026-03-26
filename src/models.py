import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# 全局默认输入时间窗口大小
DEFAULT_INPUT_WINDOW = 96

class TimeSeriesDataset(Dataset):
    """
    基于滑动窗口机制的标准时间序列数据集构建类。
    支持 pandas DataFrame 与 numpy array 格式的自动兼容与转换。
    """
    def __init__(self, X, y, input_window=DEFAULT_INPUT_WINDOW, pred_window=24, stride=1):
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y = y.values if isinstance(y, pd.DataFrame) else y
        self.input_window = input_window
        self.pred_window = pred_window
        self.stride = stride
        
        # 预先计算所有合法的滑动窗口起始索引，保证 PyTorch DataLoader 的内存访问效率
        self.indices = list(range(0, len(self.X) - input_window - pred_window + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        X_seq = self.X[start_idx : start_idx + self.input_window]
        y_seq = self.y[start_idx + self.input_window : start_idx + self.input_window + self.pred_window]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)


class TimeSeriesDatasetWithNoise(Dataset):
    """
    带有数据增强机制的数据集包装器。
    在训练模式下，通过向输入序列注入高斯噪声来提升模型对抗传感器高频毛刺的鲁棒性。
    """
    def __init__(self, dataset, noise_std=0.01):
        self.dataset = dataset
        self.noise_std = noise_std
        self.training = False 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # 仅在训练阶段且噪声标准差大于0时进行数据增强
        if self.training and self.noise_std > 0:
            noise = torch.normal(mean=0.0, std=self.noise_std, size=x.shape)
            x = x + noise
        return x, y


class BaselineMLP(nn.Module):
    """
    作为基线对照的 MLP 网络，用于验证时序特征映射的基础性能。
    加入 BatchNorm1d 与 Dropout 防止过拟合并加速收敛。
    """
    def __init__(self, input_dim):
        super(BaselineMLP, self).__init__()
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
    基于 Transformer 编码器架构的时序预测模型。
    利用自注意力机制捕获序列中的全局时间依赖关系。
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pred_window, nhead=4):
        super(TimeSeriesTransformer, self).__init__()
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
        x = self.input_projection(x)  
        out = self.transformer(x)  
        # 提取序列最后一个时间步的隐状态作为全局表征进行解码预测
        out = self.fc(out[:, -1, :])  
        return out.view(x.size(0), -1, 1)  


class LSTMGRUForecaster(nn.Module):
    """
    混合循环神经网络架构，结合 LSTM 的长记忆能力与 GRU 的高计算效率。
    引入残差连接机制缓解深层网络带来的梯度消失问题。
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
        # 跨越 RNN 层的线性残差映射，提供基准参考
        self.residual_fc = nn.Linear(input_dim, gru_hidden_dim)
        self.fc = nn.Linear(gru_hidden_dim, pred_window * output_dim)
        self.pred_window = pred_window
        self.output_dim = output_dim

    def forward(self, x):
        residual = self.residual_fc(x[:, -1, :])
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(lstm_out)
        
        last_hidden = gru_out[:, -1, :] + residual
        out = self.fc(last_hidden)
        out = out.view(x.size(0), self.pred_window, self.output_dim)
        return out


class CNNLSTMForecaster(nn.Module):
    """
    时空混合架构预测器。
    使用 1D CNN 提取多变量时间序列的局部特征，随后通过 LSTM 捕获全局时序演变规律。
    """
    def __init__(self, input_dim, lstm_hidden_dim, num_layers, output_dim, pred_window, dropout=0.2):
        super(CNNLSTMForecaster, self).__init__()
        # 连续的 1D 卷积层，用于扩张感受野并提取传感器间的局部交互特征
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
        residual = self.residual_fc(x[:, -1, :])
        
        # 置换维度以适配 Conv1d 对通道数在第二维的要求 (Batch, Channels, Length)
        x = x.transpose(1, 2)  
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # 卷积提取完毕，恢复维度顺序以输入 LSTM (Batch, Length, Channels)
        x = x.transpose(1, 2)  
        
        lstm_out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1] + residual
        
        out = self.fc(last_hidden)
        out = out.view(x.size(0), self.pred_window, self.output_dim)
        return out
