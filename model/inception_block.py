import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()
        
        self.branch1 = nn.Conv3d(in_channels, 64, kernel_size=1)
        
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, 96, kernel_size=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 128, kernel_size=3, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=5, padding=2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, 32, kernel_size=1)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], 1)