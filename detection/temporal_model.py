import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Computes learnable temporal attention weights across sequence frames.
    Allows the model to focus on decisive strike, punch, and impact moments.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, rnn_outputs: torch.Tensor):
        # rnn_outputs: (batch_size, seq_len, hidden_dim)
        scores = self.attn(rnn_outputs)  # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(rnn_outputs * weights, dim=1)  # (batch_size, hidden_dim)
        return context, weights


class FightingBiLSTM(nn.Module):
    """
    PyTorch Bi-Directional LSTM Temporal Action Classifier.
    Processes sequential MediaPipe kinematic pose features (normalized landmarks,
    velocities, joint angles, inter-person dynamics).
    """
    def __init__(
        self,
        input_dim: int = 510,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input feature projection & normalization
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 2-Layer Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # BiLSTM produces 2 * hidden_dim = 256
        self.attention = TemporalAttention(hidden_dim * 2)

        # Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, input_dim)
        x_norm = self.input_norm(x)
        x_proj = self.input_proj(x_norm)
        
        lstm_out, _ = self.lstm(x_proj)  # (batch_size, seq_len, 256)
        context, attn_weights = self.attention(lstm_out)  # (batch_size, 256)
        
        prob = self.classifier(context)  # (batch_size, 1)
        return prob, attn_weights
