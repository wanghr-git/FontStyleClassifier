import torch
import torch.nn as nn

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同


# 输入维度和输出维度不一致的卷积模块
class ConvBlock_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBnRelu(nn.Module):
    def __init__(self, ch_out):
        super(ConvBnRelu, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBn(nn.Module):
    def __init__(self, ch_out):
        super(ConvBn, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out))

    def forward(self, x):
        x = self.block(x)
        return x


# 输入维度和输出维度一致的卷积块，可以重复多个卷积块
class ConvBlock_2(nn.Module):
    def __init__(self, ch_out, t=2):
        super(ConvBlock_2, self).__init__()
        self.t = t
        self.layers = nn.ModuleList()
        for i in range(t):
            self.layers.append(ConvBnRelu(ch_out))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SwordNet(nn.Module):
    def __init__(self, in_ch, num_classes):
        nn.Module.__init__(self)
        self.Conv_Block_1 = nn.Sequential(
            ConvBlock_1(ch_in=in_ch, ch_out=64),
            ConvBlock_1(ch_in=64, ch_out=128))
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv_Block_2 = nn.Sequential(
            ConvBlock_2(ch_out=128, t=2),
            ConvBn(ch_out=128)
        )

        self.ReluMaxpool_1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv_Block_3 = nn.Sequential(
            ConvBlock_1(ch_in=128, ch_out=256),
            ConvBlock_2(ch_out=256, t=1))

        self.Conv_Block_4 = nn.Sequential(
            ConvBlock_2(ch_out=256, t=2),
            ConvBn(ch_out=256)
        )

        self.ReluMaxpool_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv_Block_5 = ConvBlock_1(ch_in=256, ch_out=512)
        self.Conv_Block_6 = nn.Sequential(
            ConvBlock_2(ch_out=512, t=2),
            ConvBn(ch_out=512)
        )

        self.ReluMaxpool_3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv_Block_7 = ConvBlock_1(ch_in=512, ch_out=1024)

        self.FC = nn.Sequential(nn.Flatten(),
                                nn.Linear(in_features=1024, out_features=num_classes),
                                nn.Softmax(dim=1))

    def forward(self, x):
        x = self.Conv_Block_1(x)
        x = self.Maxpool(x)
        x1 = self.Conv_Block_2(x)
        x = x + x1   # skip-connection
        x = self.ReluMaxpool_1(x)
        x = self.Conv_Block_3(x)
        x1 = self.Conv_Block_4(x)
        x = x + x1   # skip-connection
        x = self.ReluMaxpool_2(x)
        x = self.Conv_Block_5(x)
        x1 = self.Conv_Block_6(x)
        x = x + x1   # skip-connection
        x = self.ReluMaxpool_3(x)
        x = self.Conv_Block_7(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = nn.functional.dropout(x, p=0.5)
        x = self.FC(x)
        return x
