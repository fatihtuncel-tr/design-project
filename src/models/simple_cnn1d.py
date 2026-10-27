import torch
import torch.nn as nn

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dropout_rate=0.2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        return self.dropout(self.pool(self.relu(self.bn(self.conv(x)))))

class SimpleCNN1D(nn.Module):
    def __init__(self, in_channels=12, num_classes=5, dropout_rate=0.2):
        """
        Hafif 4-bloklu 1D-CNN modeli (~0.3M parametre).
        
        Args:
            in_channels (int): Derivasyon sayisi (12 veya 1).
            num_classes (int): Cikti sinif sayisi (PTB-XL superclasses = 5).
            dropout_rate (float): MC Dropout ve regulasyon icin dropout orani.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            ConvBlock1D(in_channels, 32, kernel_size=7, dropout_rate=dropout_rate),
            ConvBlock1D(32, 64, kernel_size=7, dropout_rate=dropout_rate),
            ConvBlock1D(64, 128, kernel_size=5, dropout_rate=dropout_rate),
            ConvBlock1D(128, 256, kernel_size=5, dropout_rate=dropout_rate),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        pooled = self.global_pool(feat).squeeze(-1)
        logits = self.classifier(pooled)
        return logits
