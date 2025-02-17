import torch
from .ghost_convolution import GhostModule
import torch.nn as nn
import math

class CrossAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 替换交叉注意力中的卷积为ghost convolution

        self.in_channels = in_channels

        self.query1 = GhostModule(in_channels, in_channels // 8)
        self.key1 = GhostModule(in_channels, in_channels // 4)
        self.value1 = GhostModule(in_channels, in_channels)

        self.query2 = GhostModule(in_channels, in_channels // 8)
        self.key2 = GhostModule(in_channels, in_channels // 4)
        self.value2 = GhostModule(in_channels, in_channels)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.conv_cat = nn.Sequential(GhostModule(in_channels * 2, out_channels, kernel_size=3),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU())  # conv_f

    def forward(self, input1, input2):
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1)
        k1 = self.key1(input1).view(batch_size, -1, height * width)
        v1 = self.value1(input1).view(batch_size, -1, height * width)

        q2 = self.query2(input2)
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        q = torch.cat([q1,q2],1).view(batch_size, -1, height * width).permute(0, 2, 1)
        attn_matrix1 = torch.bmm(q, k1)
        attn_matrix1 = self.softmax(attn_matrix1)
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1))
        out1 = out1.view(*input1.shape)
        out1 = self.gamma * out1 + input1


        attn_matrix2 = torch.bmm(q, k2)
        attn_matrix2 = self.softmax(attn_matrix2)
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))
        out2 = out2.view(*input2.shape)
        out2 = self.gamma * out2 + input2

        feat_sum = self.conv_cat(torch.cat([out1,out2],1))
        return feat_sum, out1, out2


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleLayer, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv_transpose(x)


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # 输入shape为(B, C, H, W)
        return self.conv(x)


class CA_BiFeature(nn.Module):
    def __init__(self):
        super(CA_BiFeature, self).__init__()
        self.cross1 = CrossAtt(1024, 1024)
        self.cross2 = CrossAtt(512, 512)
        self.cross3 = CrossAtt(256, 256)
        self.cross4 = CrossAtt(128, 128)

        self.upsample1 = UpsampleLayer(1024, 512)
        self.upsample2 = UpsampleLayer(512, 256)
        self.upsample3 = UpsampleLayer(256, 128)

        self.downsample1 = DownsampleLayer(128, 256)
        self.downsample2 = DownsampleLayer(256, 512)
        self.downsample3 = DownsampleLayer(512, 1024)

        self.conv1 = nn.Conv2d(in_channels=1024 * 3, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=512 * 3, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=256 * 3, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128 * 2, out_channels=128, kernel_size=1, stride=1, padding=0)

    def transpose(self, x):
        B, HW, C = x.size()
        H = int(math.sqrt(HW))     # 求平方根
        x = x.transpose(1, 2)      # （B,HW,C） -> (B, C, HW)
        x = x.view(B, C, H, H)     # (B, C, HW) -> (B, C, H, W)
        return x


    def forward(self, x_diff, x_downsample1, x_downsample2):
        x_diff[3] = self.transpose(x_diff[3])  # 4, 1024, 7, 7
        x_diff[2] = self.upsample1(self.transpose(x_diff[2]))  # 4, 512, 14, 14
        x_diff[1] = self.upsample2(self.transpose(x_diff[1]))  # 4, 256, 28, 28
        x_diff[0] = self.upsample3(self.transpose(x_diff[0]))  # 4, 128, 56, 56

        x_downsample1[3] = self.transpose(x_downsample1[3])    # 4, 1024, 7, 7
        x_downsample2[3] = self.transpose(x_downsample2[3])    # 4, 1024, 7, 7
        x_downsample1[2] = self.transpose(x_downsample1[2])    # 4, 512, 14, 14
        x_downsample2[2] = self.transpose(x_downsample2[2])    # 4, 512, 14, 14
        x_downsample1[1] = self.transpose(x_downsample1[1])    # 4, 256, 28, 28
        x_downsample2[1] = self.transpose(x_downsample2[1])    # 4, 256, 28, 28
        x_downsample1[0] = self.transpose(x_downsample1[0])    # 4, 128, 56, 56
        x_downsample2[0] = self.transpose(x_downsample2[0])    # 4, 128, 56, 56

        # 双时态特征进行交叉注意力
        bi_ca1, cao11, cao12 = self.cross1(x_downsample1[3], x_downsample2[3])   # 4, 1024, 7
        bi_ca2, cao21, cao22 = self.cross2(x_downsample1[2], x_downsample2[2])   # 4, 512, 14, 14
        bi_ca3, cao31, cao32 = self.cross3(x_downsample1[1], x_downsample2[1])   # 4, 256, 28, 28
        bi_ca4, cao41, cao42 = self.cross4(x_downsample1[0], x_downsample2[0])

        x_diff1 = cao11 - cao12
        x_diff2 = cao21 - cao22
        x_diff3 = cao31 - cao32
        x_diff4 = cao41 - cao42

        cat32_1 = self.downsample3(x_downsample1[2])
        cat32_2 = self.downsample3(x_downsample2[2])
        cat21_1 = self.downsample2(x_downsample1[1])
        cat21_2 = self.downsample2(x_downsample2[1])
        cat10_1 = self.downsample1(x_downsample1[0])
        cat10_2 = self.downsample1(x_downsample2[0])

        # out1 = self.conv1(torch.cat((x_diff[3], bi_ca1, cat32_1 + cat32_2), dim=1)) + x_diff[3]
        # out2 = self.conv2(torch.cat((x_diff[2], bi_ca2, cat21_1 + cat21_2), dim=1)) + x_diff[2]
        # out3 = self.conv3(torch.cat((x_diff[1], bi_ca3, cat10_1 + cat10_2), dim=1)) + x_diff[1]
        # out4 = self.conv4(torch.cat((x_diff[0], bi_ca4), dim=1)) + x_diff[0]
        out1 = self.conv1(torch.cat((x_diff1, bi_ca1, cat32_1 + cat32_2), dim=1)) + x_diff1
        out2 = self.conv2(torch.cat((x_diff2, bi_ca2, cat21_1 + cat21_2), dim=1)) + x_diff2
        out3 = self.conv3(torch.cat((x_diff3, bi_ca3, cat10_1 + cat10_2), dim=1)) + x_diff3
        out4 = self.conv4(torch.cat((x_diff4, bi_ca4), dim=1)) + x_diff4
        # summation fusion
        # out1 = x_diff[3] + bi_ca1 + (cat32_1 + cat32_2)
        # out2 = x_diff[2] + bi_ca2 + (cat21_1 + cat21_2)
        # out3 = x_diff[1] + bi_ca3 + (cat10_1 + cat10_2)
        # out4 = x_diff[0] + bi_ca4
        # no adjacent feature fusion
        # out1 = self.conv1(torch.cat((x_diff[3], bi_ca1), dim=1)) + x_diff[3]
        # out2 = self.conv2(torch.cat((x_diff[2], bi_ca2), dim=1)) + x_diff[2]
        # out3 = self.conv3(torch.cat((x_diff[1], bi_ca3), dim=1)) + x_diff[1]
        # out4 = self.conv4(torch.cat((x_diff[0], bi_ca4), dim=1)) + x_diff[0]
        # STCA abaltion
        # out1 = self.conv1(bi_ca1)
        # out2 = self.conv2(bi_ca2)
        # out3 = self.conv3(bi_ca3)
        # out4 = self.conv4(bi_ca4)

        # without JointAtt
        # out1 = self.conv1(torch.cat((x_diff[3], x_downsample1[3] + x_downsample2[3], cat32_1 + cat32_2), dim=1)) + x_diff[3]
        # out2 = self.conv2(torch.cat((x_diff[2], x_downsample1[2] + x_downsample2[2], cat21_1 + cat21_2), dim=1)) + x_diff[2]
        # out3 = self.conv3(torch.cat((x_diff[1], x_downsample1[1] + x_downsample2[1], cat10_1 + cat10_2), dim=1)) + x_diff[1]
        # out4 = self.conv4(torch.cat((x_diff[0], x_downsample1[0] + x_downsample2[0]), dim=1)) + x_diff[0]

        return out1, out2, out3, out4

