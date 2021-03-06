"""
Based on:
https://pdfs.semanticscholar.org/ca76/9bc02cb1b74a160d606fbb171afb13d0d615.pdf
"""

import torch.nn as nn
import torch.nn.functional as F


class FCN_net1(nn.Module):
    def __init__(self):
        super(FCN_net1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):
        output = self.relu1(self.bn1(self.conv1(inp)))
        output = self.relu2(self.bn2(self.conv2(output)))
        output = self.relu3(self.bn3(self.conv3(output)))
        output = self.relu4(self.bn4(self.conv4(output)))

        output = self.conv5(output)

        return output


class FCN_net2(nn.Module):  # Upgrade A
    def __init__(self):
        super(FCN_net2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):
        output = self.relu1(self.bn1(self.conv1(inp)))
        output = self.relu2(self.bn2(self.conv2(output)))
        output = self.relu3(self.bn3(self.conv3(output)))
        output = self.relu4(self.bn4(self.conv4(output)))
        output = self.relu5(self.bn5(self.conv5(output)))

        output = self.conv6(output)

        return output


class FCN_net3(nn.Module):  # Upgrade B
    def __init__(self):
        super(FCN_net3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):
        output = self.relu1(self.bn1(self.conv1(inp)))
        output = self.relu2(self.bn2(self.conv2(output)))
        output = self.relu3(self.bn3(self.conv3(output)))
        output = self.relu4(self.bn4(self.conv4(output)))
        output = self.relu5(self.bn5(self.conv5(output)))

        output = self.conv6(output)

        return output


class FCN_net4(nn.Module):  # Upgrade C
    def __init__(self):
        super(FCN_net4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        # T?? warst?? mo??na jeszcze zmieni?? na kernel_size=3
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        # T?? warst?? mo??na jeszcze poprzedzi?? tak?? sam??
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):
        output = self.relu1(self.bn1(self.conv1(inp)))
        output = self.relu2(self.bn2(self.conv2(output)))
        output = self.relu3(self.bn3(self.conv3(output)))
        output = self.relu4(self.bn4(self.conv4(output)))
        output = self.relu5(self.bn5(self.conv5(output)))

        output = self.conv6(output)

        return output


class FCN_net5(nn.Module):  # Upgrades A, B, C
    def __init__(self):
        super(FCN_net5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn8 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):
        output = self.relu1(self.bn1(self.conv1(inp)))
        output = self.relu2(self.bn2(self.conv2(output)))
        output = self.relu3(self.bn3(self.conv3(output)))
        output = self.relu4(self.bn4(self.conv4(output)))
        output = self.relu5(self.bn5(self.conv5(output)))
        output = self.relu6(self.bn6(self.conv6(output)))
        output = self.relu7(self.bn7(self.conv7(output)))
        output = self.relu8(self.bn8(self.conv8(output)))

        output = self.conv9(output)

        return output


class FCN_net_mega(nn.Module):  # Upgrades AA, BB, CC
    def __init__(self):
        super(FCN_net_mega, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn10 = nn.BatchNorm2d(32)
        self.relu10 = nn.ReLU()

        self.conv11 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn11 = nn.BatchNorm2d(32)
        self.relu11 = nn.ReLU()

        self.conv12 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):
        output = self.relu1(self.bn1(self.conv1(inp)))
        output = self.relu2(self.bn2(self.conv2(output)))
        output = self.relu3(self.bn3(self.conv3(output)))
        output = self.relu4(self.bn4(self.conv4(output)))
        output = self.relu5(self.bn5(self.conv5(output)))
        output = self.relu6(self.bn6(self.conv6(output)))
        output = self.relu7(self.bn7(self.conv7(output)))
        output = self.relu8(self.bn8(self.conv8(output)))
        output = self.relu9(self.bn9(self.conv9(output)))
        output = self.relu10(self.bn10(self.conv10(output)))
        output = self.relu11(self.bn11(self.conv11(output)))

        output = self.conv12(output)

        return output


class FCN_net_mega_V2(nn.Module):
    def __init__(self):
        super(FCN_net_mega_V2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn10 = nn.BatchNorm2d(32)
        self.relu10 = nn.ReLU()

        self.conv11 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn11 = nn.BatchNorm2d(32)
        self.relu11 = nn.ReLU()

        self.conv12 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):

        output = self.bn1(self.relu1(self.conv1(inp)))
        output = self.bn2(self.relu2(self.conv2(output)))
        output = self.bn3(self.relu3(self.conv3(output)))
        output = self.bn4(self.relu4(self.conv4(output)))
        output = self.bn5(self.relu5(self.conv5(output)))
        output = self.bn6(self.relu6(self.conv6(output)))
        output = self.bn7(self.relu7(self.conv7(output)))
        output = self.bn8(self.relu8(self.conv8(output)))
        output = self.bn9(self.relu9(self.conv9(output)))
        output = self.bn10(self.relu10(self.conv10(output)))
        output = self.bn11(self.relu11(self.conv11(output)))

        output = self.conv12(output)

        return output


class FCN_net_mega_dropout(nn.Module):
    def __init__(self):
        super(FCN_net_mega_dropout, self).__init__()
        dropout_rate = 0.2
        self.conv2_drop = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn10 = nn.BatchNorm2d(32)
        self.relu10 = nn.ReLU()

        self.conv11 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn11 = nn.BatchNorm2d(32)
        self.relu11 = nn.ReLU()

        self.conv12 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):

        output = self.conv2_drop(self.bn1(self.relu1(self.conv1(inp))))
        output = self.conv2_drop(self.bn2(self.relu2(self.conv2(output))))
        output = self.conv2_drop(self.bn3(self.relu3(self.conv3(output))))
        output = self.conv2_drop(self.bn4(self.relu4(self.conv4(output))))
        output = self.conv2_drop(self.bn5(self.relu5(self.conv5(output))))
        output = self.conv2_drop(self.bn6(self.relu6(self.conv6(output))))
        output = self.conv2_drop(self.bn7(self.relu7(self.conv7(output))))
        output = self.conv2_drop(self.bn8(self.relu8(self.conv8(output))))
        output = self.conv2_drop(self.bn9(self.relu9(self.conv9(output))))
        output = self.conv2_drop(self.bn10(self.relu10(self.conv10(output))))
        output = self.conv2_drop(self.bn11(self.relu11(self.conv11(output))))

        output = self.conv12(output)

        return output


class FCN_net_mega_dropout2(nn.Module):
    def __init__(self):
        super(FCN_net_mega_dropout2, self).__init__()
        dropout_rate = 0.5
        self.conv2_drop = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn10 = nn.BatchNorm2d(32)
        self.relu10 = nn.ReLU()

        self.conv11 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn11 = nn.BatchNorm2d(32)
        self.relu11 = nn.ReLU()

        self.conv12 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):

        output = self.conv2_drop(self.bn1(self.relu1(self.conv1(inp))))
        output = self.conv2_drop(self.bn2(self.relu2(self.conv2(output))))
        output = self.conv2_drop(self.bn3(self.relu3(self.conv3(output))))
        output = self.conv2_drop(self.bn4(self.relu4(self.conv4(output))))
        output = self.conv2_drop(self.bn5(self.relu5(self.conv5(output))))
        output = self.conv2_drop(self.bn6(self.relu6(self.conv6(output))))
        output = self.conv2_drop(self.bn7(self.relu7(self.conv7(output))))
        output = self.conv2_drop(self.bn8(self.relu8(self.conv8(output))))
        output = self.conv2_drop(self.bn9(self.relu9(self.conv9(output))))
        output = self.conv2_drop(self.bn10(self.relu10(self.conv10(output))))
        output = self.conv2_drop(self.bn11(self.relu11(self.conv11(output))))

        output = self.conv12(output)

        return output


class FCN_net_mega_dropout3(nn.Module):
    def __init__(self):
        super(FCN_net_mega_dropout3, self).__init__()
        dropout_rate = 0.3
        self.conv2_drop = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn10 = nn.BatchNorm2d(32)
        self.relu10 = nn.ReLU()

        self.conv11 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn11 = nn.BatchNorm2d(32)
        self.relu11 = nn.ReLU()

        self.conv12 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):

        output = self.conv2_drop(self.bn1(self.relu1(self.conv1(inp))))
        output = self.conv2_drop(self.bn2(self.relu2(self.conv2(output))))
        output = self.conv2_drop(self.bn3(self.relu3(self.conv3(output))))
        output = self.conv2_drop(self.bn4(self.relu4(self.conv4(output))))
        output = self.conv2_drop(self.bn5(self.relu5(self.conv5(output))))
        output = self.conv2_drop(self.bn6(self.relu6(self.conv6(output))))
        output = self.conv2_drop(self.bn7(self.relu7(self.conv7(output))))
        output = self.conv2_drop(self.bn8(self.relu8(self.conv8(output))))
        output = self.conv2_drop(self.bn9(self.relu9(self.conv9(output))))
        output = self.conv2_drop(self.bn10(self.relu10(self.conv10(output))))
        output = self.conv2_drop(self.bn11(self.relu11(self.conv11(output))))

        output = self.conv12(output)

        return output


class FCN_net_mega_sigmoid(nn.Module):
    def __init__(self):
        super(FCN_net_mega_sigmoid, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.sigm1 = nn.Sigmoid()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.sigm2 = nn.Sigmoid()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.sigm3 = nn.Sigmoid()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.sigm4 = nn.Sigmoid()

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.sigm5 = nn.Sigmoid()

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.sigm6 = nn.Sigmoid()

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.sigm7 = nn.Sigmoid()

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.sigm8 = nn.Sigmoid()

        self.conv9 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.sigm9 = nn.Sigmoid()

        self.conv10 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn10 = nn.BatchNorm2d(32)
        self.sigm10 = nn.Sigmoid()

        self.conv11 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn11 = nn.BatchNorm2d(32)
        self.sigm11 = nn.Sigmoid()

        self.conv12 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):

        output = self.bn1(self.sigm1(self.conv1(inp)))
        output = self.bn2(self.sigm2(self.conv2(output)))
        output = self.bn3(self.sigm3(self.conv3(output)))
        output = self.bn4(self.sigm4(self.conv4(output)))
        output = self.bn5(self.sigm5(self.conv5(output)))
        output = self.bn6(self.sigm6(self.conv6(output)))
        output = self.bn7(self.sigm7(self.conv7(output)))
        output = self.bn8(self.sigm8(self.conv8(output)))
        output = self.bn9(self.sigm9(self.conv9(output)))
        output = self.bn10(self.sigm10(self.conv10(output)))
        output = self.bn11(self.sigm11(self.conv11(output)))

        output = self.conv12(output)

        return output
