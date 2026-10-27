"""
SentinelVision AI - Hybrid Deep Learning Architecture
MobileNetV2 Spatial Feature Extractor + BiLSTM Temporal Sequence Modeler + Attention Mechanism.

Designed for end-to-end spatiotemporal action recognition and suspicious activity detection in video feeds.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Dict, Any, Optional


class TemporalAttention(nn.Module):
    """
    Computes learnable temporal attention weights across video sequence frames.
    Directs the model's focus to pivotal motion moments (striking, sudden falls, firearm brandishing).
    """
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, rnn_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # rnn_outputs: (batch_size, seq_len, hidden_dim)
        scores = self.attn(rnn_outputs)  # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(rnn_outputs * weights, dim=1)  # (batch_size, hidden_dim)
        return context, weights


class MobileNetV2SpatialExtractor(nn.Module):
    """
    MobileNetV2 Backbone for High-Efficiency Spatial Feature Extraction.
    Uses Inverted Residual blocks with linear bottlenecks.
    Produces a compact feature embedding per video frame.
    """
    def __init__(self, feature_dim: int = 256, pretrained: bool = False):
        super(MobileNetV2SpatialExtractor, self).__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        base_mobilenet = models.mobilenet_v2(weights=weights)
        self.features = base_mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # MobileNetV2 feature channels = 1280
        self.projection = nn.Sequential(
            nn.Linear(1280, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 3, H, W)
        feat = self.features(x)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)
        feat = self.projection(feat)
        return feat


class MobileNetV2BiLSTMAttention(nn.Module):
    """
    Canonical Vision Hybrid Architecture:
    1. Spatial Feature Extractor: MobileNetV2 (Frame -> 256-dim feature embedding).
    2. Temporal Sequence Modeler: 2-layer Bidirectional LSTM (256-dim input -> 2x128 = 256-dim hidden states).
    3. Temporal Attention: Learns frame-importance distribution across sequence length T.
    4. Classification Head: Fully-connected layers for suspicious activity detection.
    """
    def __init__(
        self,
        num_classes: int = 2,
        spatial_feature_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_backbone: bool = False
    ):
        super(MobileNetV2BiLSTMAttention, self).__init__()
        self.num_classes = num_classes
        self.spatial_feature_dim = spatial_feature_dim
        self.hidden_dim = hidden_dim

        # 1. Spatial Extractor
        self.spatial_extractor = MobileNetV2SpatialExtractor(
            feature_dim=spatial_feature_dim,
            pretrained=pretrained_backbone
        )

        # 2. Temporal BiLSTM
        self.bilstm = nn.LSTM(
            input_size=spatial_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 3. Temporal Attention (bidirectional hidden dim = hidden_dim * 2)
        self.attention = TemporalAttention(hidden_dim * 2)

        # 4. Classification Head
        out_features = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, out_features)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Video tensor of shape (batch_size, seq_len, 3, H, W)
               or precomputed spatial features (batch_size, seq_len, spatial_feature_dim)
        Returns:
            logits/probs: (batch_size, num_classes) or (batch_size, 1)
            attention_weights: (batch_size, seq_len, 1)
        """
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x_flat = x.view(b * t, c, h, w)
            spatial_feats = self.spatial_extractor(x_flat)
            spatial_feats = spatial_feats.view(b, t, -1)
        else:
            # Pre-extracted frame features
            spatial_feats = x

        lstm_out, _ = self.bilstm(spatial_feats)
        context, attn_weights = self.attention(lstm_out)
        out = self.classifier(context)

        if self.num_classes == 2:
            probs = torch.sigmoid(out)
        else:
            probs = F.softmax(out, dim=-1)

        return probs, attn_weights

    def get_model_statistics(self) -> Dict[str, Any]:
        """Computes exact parameter counts and memory footprint."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        param_size_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        model_size_mb = (param_size_bytes + buffer_size_bytes) / (1024 * 1024)

        return {
            "model_name": "MobileNetV2 + BiLSTM + Attention",
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "model_size_mb": round(float(model_size_mb), 2),
            "backbone": "MobileNetV2 (1280 -> 256)",
            "temporal_model": f"{self.bilstm.num_layers}-Layer BiLSTM (hidden={self.hidden_dim})",
            "attention_mechanism": "Additive Self-Attention (Tanh + Softmax)",
            "classes": self.num_classes
        }


if __name__ == "__main__":
    model = MobileNetV2BiLSTMAttention(num_classes=2)
    stats = model.get_model_statistics()
    print("MobileNetV2 + BiLSTM + Attention Model Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
