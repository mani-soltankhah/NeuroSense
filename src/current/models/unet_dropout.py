import torch
import torch.nn as nn


class DoubleConvDropout(nn.Module):
    """Double convolution with dropout for regularization"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super(DoubleConvDropout, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUnetDropout(nn.Module):
    """Attention U-Net with Dropout"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super(AttentionUnetDropout, self).__init__()
        self.down1 = DoubleConvDropout(in_channels, 64, dropout_rate)
        self.down2 = DoubleConvDropout(64, 128, dropout_rate)
        self.down3 = DoubleConvDropout(128, 256, dropout_rate)
        self.down4 = DoubleConvDropout(256, 512, dropout_rate)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConvDropout(512, 1024, dropout_rate)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(512, 512, 256)
        self.conv4 = DoubleConvDropout(1024, 512, dropout_rate)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(256, 256, 128)
        self.conv3 = DoubleConvDropout(512, 256, dropout_rate)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(128, 128, 64)
        self.conv2 = DoubleConvDropout(256, 128, dropout_rate)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(64, 64, 32)
        self.conv1 = DoubleConvDropout(128, 64, dropout_rate)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        x5 = self.bottleneck(self.pool(x4))

        d4 = self.up4(x5)
        x4 = self.att4(g=d4, x=x4)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.conv4(d4)

        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=x3)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.conv3(d3)

        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=x2)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv2(d2)

        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=x1)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.conv1(d1)

        return self.final(d1)


class AttentionUnetLarge(nn.Module):
    """Larger Attention U-Net with 2x channels"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super(AttentionUnetLarge, self).__init__()
        self.down1 = DoubleConvDropout(in_channels, 128, dropout_rate)  # 64 → 128
        self.down2 = DoubleConvDropout(128, 256, dropout_rate)  # 128 → 256
        self.down3 = DoubleConvDropout(256, 512, dropout_rate)  # 256 → 512
        self.down4 = DoubleConvDropout(512, 1024, dropout_rate)  # 512 → 1024
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConvDropout(1024, 2048, dropout_rate)  # 1024 → 2048

        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.att4 = AttentionGate(1024, 1024, 512)
        self.conv4 = DoubleConvDropout(2048, 1024, dropout_rate)

        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att3 = AttentionGate(512, 512, 256)
        self.conv3 = DoubleConvDropout(1024, 512, dropout_rate)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionGate(256, 256, 128)
        self.conv2 = DoubleConvDropout(512, 256, dropout_rate)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att1 = AttentionGate(128, 128, 64)
        self.conv1 = DoubleConvDropout(256, 128, dropout_rate)

        self.final = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        x5 = self.bottleneck(self.pool(x4))

        d4 = self.up4(x5)
        x4 = self.att4(g=d4, x=x4)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.conv4(d4)

        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=x3)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.conv3(d3)

        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=x2)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv2(d2)

        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=x1)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.conv1(d1)

        return self.final(d1)
