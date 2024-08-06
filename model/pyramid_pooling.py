import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels=1024, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPooling, self).__init__()
        
        self.paths = nn.ModuleList()
        for pool_size in pool_sizes:
            self.paths.append(nn.Sequential(
                nn.AdaptiveAvgPool3d(pool_size),
                nn.Conv3d(in_channels, out_channels // len(pool_sizes), kernel_size=1),
                nn.BatchNorm3d(out_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            ))
        
    def forward(self, x):
        size = x.size()
        out = [x]
        for path in self.paths:
            out.append(F.interpolate(path(x), size[2:], mode='trilinear', align_corners=True))
        return torch.cat(out, 1)