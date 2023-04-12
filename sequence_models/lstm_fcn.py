import torch
import torch.nn as nn

class LSTM_FCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(LSTM_FCN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]

        x = x.permute(0, 2, 1)
        conv_out = self.conv1(x)
        conv_out = self.conv2(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = conv_out.mean(2)

        combined = lstm_out + conv_out
        out = self.fc(combined)
        return out
