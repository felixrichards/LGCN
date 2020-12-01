import torch.nn as nn
from lgcn.cmplx_modules import (
    LGConvCmplx,
    ReLUCmplx,
    BatchNormCmplx,
    AvgPoolCmplx,
    Project
)
from lgcn.cmplx import new_cmplx


class LinearBlock(nn.Module):
    def __init__(self, fcn, dropout=0., **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(fcn, fcn),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.block(x)


class LGCN(nn.Module):
    def __init__(self, n_channels, n_classes, no_g=4):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            LGConvCmplx(n_channels, 16, 3, no_g=no_g, padding=1),
            LGConvCmplx(16, 32, 3, no_g=no_g, gabor_pooling='max'),
            AvgPoolCmplx(kernel_size=2, stride=2),
            ReLUCmplx(inplace=True, channels=32),
            BatchNormCmplx(32),
        )
        self.conv_block2 = nn.Sequential(
            LGConvCmplx(32, 32, 3, no_g=no_g, padding=1),
            LGConvCmplx(32, 64, 3, no_g=no_g, gabor_pooling='max'),
            AvgPoolCmplx(kernel_size=2, stride=2),
            ReLUCmplx(inplace=True, channels=64),
            BatchNormCmplx(64),
        )
        self.conv_block3 = nn.Sequential(
            LGConvCmplx(64, 64, 3, no_g=no_g, padding=1),
            LGConvCmplx(64, 128, 3, no_g=no_g, gabor_pooling='max'),
            AvgPoolCmplx(kernel_size=2, stride=1),
            ReLUCmplx(inplace=True, channels=128),
            BatchNormCmplx(128),
        )
        self.project = Project()
        self.fcn = 128 * 2 * 2 * 2
        self.linear = nn.Sequential(
            LinearBlock(self.fcn),
            LinearBlock(self.fcn),
            LinearBlock(self.fcn)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.fcn, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = new_cmplx(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.project(x)
        x = x.flatten(1)
        x = self.linear(x)
        x = self.classifier(x)
        return x
