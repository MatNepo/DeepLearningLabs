import torch
import torch.nn as nn

class LMUModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LMUModel, self).__init__()
        self.lmu = nn.LSTM(input_size, 128)  # Пример LSTM, замените на свою LMU
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        out, _ = self.lmu(x)
        out = self.fc(out[:, -1, :])
        return out
