import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=6,
            kernel_size=5,
            padding=2
        )
        self.sigmoid1 = nn.Sigmoid()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5

        )
        self.sigmoid2 = nn.Sigmoid()
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.convlayer = nn.Sequential(
            self.conv1,
            self.sigmoid1,
            self.avgpool1,
            self.conv2,
            self.sigmoid2,
            self.avgpool2
        )

        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.sigmoid3 = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.sigmoid4 = nn.Sigmoid()
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

        self.fullyconnected = nn.Sequential(
            self.fc1, self.sigmoid3, self.fc2, self.sigmoid4, self.fc3
        )

    def forward(self, x):
        x = self.convlayer(x)
        x = x.view(x.shape[0], -1)
        x = self.fullyconnected(x)
        return x
    
    def layer_summary(self, x):
        for layer in self.convlayer:
            x = layer(x)
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.AvgPool2d)):
                print(f"{layer.__class__.__name__}: {x.shape}")
        x = x.view(x.shape[0], -1)
        for layer in self.fullyconnected:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                print(f"{layer.__class__.__name__}: {x.shape}")