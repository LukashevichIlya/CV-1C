import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, padding=1):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=channels)

        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                               kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=channels)

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=self.expansion * channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.expansion * channels)

        if stride != 1 or in_channels != self.expansion * channels:
            self.res_connection = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=self.expansion * channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=self.expansion * channels)
            )
        else:
            self.res_connection = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, src):
        out = self.relu(self.bn1(self.conv1(src)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.res_connection is not None:
            out += self.res_connection(src)
        else:
            out += src

        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, block, num_classes):
        super(ResNet50, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.in_channels)

        self.block_layer1 = self._make_layer(block, channels=64, num_blocks=3, stride=1)
        self.block_layer2 = self._make_layer(block, channels=128, num_blocks=4, stride=2)
        self.block_layer3 = self._make_layer(block, channels=256, num_blocks=6, stride=2)
        self.block_layer4 = self._make_layer(block, channels=512, num_blocks=3, stride=2)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.relu = nn.ReLU()

        self.fc = nn.Linear(in_features=512 * block.expansion, out_features=num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, src):
        out = self.max_pool(self.relu(self.bn1(self.conv1(src))))
        out = self.block_layer1(out)
        out = self.block_layer2(out)
        out = self.block_layer3(out)
        out = self.block_layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
