import math
import torch.nn as nn

import curves

__all__ = [
    'BasicCNN',
]


class BasicCNNBase(nn.Module):

    def __init__(self, num_classes):
        super(BasicCNNBase, self).__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3),),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),),
            nn.Flatten()
        )
        
        self.fc_part = nn.Sequential(
            nn.Linear(in_features=576, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10)
        )

        # Initialize weights
        for m in self.conv_part.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_part(x)
        x = x.view(x.size(0), -1)
        x = self.fc_part(x)
        return x


class BasicCNNCurve(nn.Module):
    def __init__(self, num_classes, fix_points):
        super(BasicCNNCurve, self).__init__()
        self.conv1 = curves.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), fix_points=fix_points)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = curves.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), fix_points=fix_points)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = curves.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), fix_points=fix_points)
        self.relu3 = nn.ReLU()
        self.flatten1 = nn.Flatten()

        self.fc4 = curves.Linear(in_features=576, out_features=64, fix_points=fix_points)
        self.relu4 = nn.ReLU()
        self.fc5 = curves.Linear(in_features=64, out_features=64, fix_points=fix_points)
        self.relu5 = nn.ReLU()
        self.fc6 = curves.Linear(in_features=64, out_features=10, fix_points=fix_points)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, curves.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2. / n))
                    getattr(m, 'bias_%d' % i).data.zero_()

    def forward(self, x, coeffs_t):
        x = self.conv1(x, coeffs_t)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.conv2(x, coeffs_t)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = self.conv3(x, coeffs_t)
        x = self.relu3(x)
        x = self.flatten1(x)

        x = x.view(x.size(0), -1)

        x = self.fc4(x, coeffs_t)
        x = self.relu4(x)

        x = self.fc5(x, coeffs_t)
        x = self.relu5(x)

        x = self.fc6(x, coeffs_t)
        
        return x


class BasicCNN:
    base = BasicCNNBase
    curve = BasicCNNCurve
    kwargs = {}
