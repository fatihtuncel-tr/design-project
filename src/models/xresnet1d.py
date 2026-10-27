import torch
import torch.nn as nn

class ConvLayer1D(nn.Module):
    def __init__(self, ni, nf, ks=3, stride=1, act=True):
        super().__init__()
        self.conv = nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=ks // 2, bias=False)
        self.bn = nn.BatchNorm1d(nf)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock1D(nn.Module):
    def __init__(self, ni, nf, stride=1, ks=5, dropout_rate=0.2):
        super().__init__()
        self.conv1 = ConvLayer1D(ni, nf, ks=ks, stride=stride, act=True)
        self.conv2 = ConvLayer1D(nf, nf, ks=ks, stride=1, act=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.act = nn.ReLU(inplace=True)

        if ni != nf or stride != 1:
            self.shortcut = ConvLayer1D(ni, nf, ks=1, stride=stride, act=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv2(self.conv1(x))
        out = self.dropout(out)
        return self.act(out + res)

class xResNet1D(nn.Module):
    def __init__(self, in_channels=12, num_classes=5, layers=[2, 2, 2, 2], nf=32, dropout_rate=0.2):
        """
        xResNet1d mimarisi (PTB-XL kıyaslama standardı).
        
        Args:
            in_channels (int): Derivasyon sayisi (12 veya 1).
            num_classes (int): Sinif sayisi (5 super-class).
            layers (list): Her asamadaki residual blok sayisi.
            nf (int): Temel filtre sayisi (varsayilan 32).
            dropout_rate (float): Dropout orani (MC Dropout icin).
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Giris stem'i (3 asamali konvolusyon)
        self.stem = nn.Sequential(
            ConvLayer1D(in_channels, nf, ks=7, stride=2),
            ConvLayer1D(nf, nf, ks=3, stride=1),
            ConvLayer1D(nf, nf * 2, ks=3, stride=1),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        curr_ni = nf * 2
        stages = []
        for i, num_blocks in enumerate(layers):
            stage_nf = nf * (2 ** (i + 1))
            stride = 1 if i == 0 else 2
            stage = [ResBlock1D(curr_ni, stage_nf, stride=stride, dropout_rate=dropout_rate)]
            for _ in range(num_blocks - 1):
                stage.append(ResBlock1D(stage_nf, stage_nf, stride=1, dropout_rate=dropout_rate))
            stages.append(nn.Sequential(*stage))
            curr_ni = stage_nf

        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(curr_ni, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

def xresnet1d101(in_channels=12, num_classes=5, dropout_rate=0.2):
    """PTB-XL makalesinde onerilen xResNet1d-101 mimarisi"""
    return xResNet1D(in_channels=in_channels, num_classes=num_classes, layers=[3, 4, 23, 3], nf=32, dropout_rate=dropout_rate)

def xresnet1d50(in_channels=12, num_classes=5, dropout_rate=0.2):
    """Daha hizli ve dengeli xResNet1d-50 mimarisi"""
    return xResNet1D(in_channels=in_channels, num_classes=num_classes, layers=[3, 4, 6, 3], nf=32, dropout_rate=dropout_rate)
