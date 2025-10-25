import torch
import torch.nn as nn

class SpeechModel(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=512, vocab_size=30, num_layers=3):
        super().__init__()

        # CNN front-end for temporal compression
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=(input_dim // 4) * 64 // 64,  # dynamic flattening
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.fc = nn.Linear(hidden_dim * 2, vocab_size + 1)  # +1 for blank
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, lengths):
        # x: (B, T, F)
        x = x.unsqueeze(1)  # add channel: (B, 1, T, F)
        x = self.conv(x)  # (B, C, T', F')
        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return self.log_softmax(x)
