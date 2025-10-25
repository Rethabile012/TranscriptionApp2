import torch
import torch.nn as nn

class SpeechModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, vocab_size=30, num_layers=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # After conv, C=64, F=input_dim//4 (approx)
        lstm_input_size = 64 * (input_dim // 4)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, vocab_size + 1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, lengths):
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.conv(x)    # (B, C, T', F')
        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return self.log_softmax(x)
