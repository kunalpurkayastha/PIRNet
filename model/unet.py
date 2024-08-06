import torch
import torch.nn as nn
from .inception_block import InceptionBlock
from .residual_block import ResidualBlock
from .pyramid_pooling import PyramidPooling

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = InceptionBlock(in_channels)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = InceptionBlock(256)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = InceptionBlock(256)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = InceptionBlock(256)
        self.pool4 = nn.MaxPool3d(2)
        
        # Bridge
        self.bridge = ResidualBlock(256)
        
        # Decoder
        self.up4 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.dec4 = InceptionBlock(512)
        self.up3 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.dec3 = InceptionBlock(512)
        self.up2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.dec2 = InceptionBlock(512)
        self.up1 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.dec1 = InceptionBlock(512)
        
        self.final = nn.Conv3d(256, out_channels, kernel_size=1)
        
        self.pyramid_pooling = PyramidPooling(out_channels)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bridge
        bridge = self.bridge(self.pool4(enc4))
        
        # Decoder
        dec4 = self.dec4(torch.cat([self.up4(bridge), enc4], 1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], 1))
        
        out = self.final(dec1)
        
        out = self.pyramid_pooling(out)
        
        return out