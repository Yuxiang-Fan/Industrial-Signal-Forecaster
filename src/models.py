import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

input_window = 96


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, input_window=input_window, pred_window=24, stride=1):
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y = y.values if isinstance(y, pd.DataFrame) else y
        self.input_window = input_window
        self.pred_window = pred_window
        self.stride = stride
        self.indices = list(range(0, len(self.X) - input_window - pred_window + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        X = self.X[i:i + self.input_window]
        y = self.y[i + self.input_window:i + self.input_window + self.pred_window]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TimeSeriesDatasetWithNoise(Dataset):
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


def KalMan(input_KALMAN):
    '''
    这里我乱写的虚构的方法
    input_KALMAN是array，shape=（24,4），表示24个时间步的4个模型预测结果
    '''
    ratio = np.random.random(4)
    return (ratio * input_KALMAN).sum(axis=1)


class NeuralNetwork(nn.Module):
    '''
    机器学习模型的标准接口形式
    '''

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
    '''
    时间序列模型的标准接口形式
    其中pred_window=48
    '''

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pred_window, nhead=4):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim * pred_window)

    def forward(self, x):
        x = self.input_projection(x)  # [batch_size, seq_len, input_dim] -> [batch_size, seq_len, hidden_dim]
        out = self.transformer(x)  # [batch_size, seq_len, hidden_dim]
        out = self.fc(out[:, -1, :])  # Use last output
        return out.view(x.size(0), -1, 1)  # [batch_size, pred_window, 1]


class LSTMGRUForecaster(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, gru_hidden_dim, lstm_layers, gru_layers, output_dim, pred_window,
                 dropout=0.15):
        super(LSTMGRUForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.gru = nn.GRU(input_size=lstm_hidden_dim, hidden_size=gru_hidden_dim, num_layers=gru_layers,
                          batch_first=True, dropout=dropout if gru_layers > 1 else 0.0)
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
    def __init__(self, input_dim, lstm_hidden_dim, num_layers, output_dim, pred_window, dropout=0.2):
        super(CNNLSTMForecaster, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.residual_fc = nn.Linear(input_dim, lstm_hidden_dim)
        self.fc = nn.Linear(lstm_hidden_dim, pred_window * output_dim)
        self.pred_window = pred_window
        self.output_dim = output_dim

    def forward(self, x):
        residual = self.residual_fc(x[:, -1, :])
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.transpose(1, 2)  # (batch, seq_len, 128)
        lstm_out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1] + residual
        out = self.fc(last_hidden)
        out = out.view(x.size(0), self.pred_window, self.output_dim)
        return out