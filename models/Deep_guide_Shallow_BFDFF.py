import math
import torch
import torch.nn as nn
from .ghost_convolution import GhostModule

class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # 输入shape为(B, C, H, W)
        return self.conv(x)


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleLayer, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv_transpose(x)


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


class bfdff(nn.Module):
    def __init__(self):
        super(bfdff, self).__init__()
        self.cross1 = CrossAtt(512, 512)
        self.cross2 = CrossAtt(1024, 1024)
        self.cross3 = CrossAtt(256, 256)

        self.downsample1 = DownsampleLayer(128, 256)
        self.downsample2 = DownsampleLayer(256, 512)
        self.downsample3 = DownsampleLayer(512, 1024)

        self.upsample1 = UpsampleLayer(1024, 512)
        self.upsample2 = UpsampleLayer(512, 256)
        self.upsample3 = UpsampleLayer(256, 128)

        # 特征融合方式：沿通道进行cat，然后用卷积处理得到全局信息=================================================
        self.conv1 = nn.Conv2d(in_channels=1024 * 3, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=512 * 3, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=256 * 3, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128 * 2, out_channels=128, kernel_size=1, stride=1, padding=0)
        # 特征融合方式：沿通道进行cat，然后用卷积处理得到全局信息=================================================


    def transpose(self, x):
        B, HW, C = x.size()
        H = int(math.sqrt(HW))     # 求平方根
        x = x.transpose(1, 2)      # （B,HW,C） -> (B, C, HW)
        x = x.view(B, C, H, H)     # (B, C, HW) -> (B, C, H, W)
        return x

    def forward(self, x_diff, x_downsample1, x_downsample2):
        sub_fe0 = self.transpose(x_diff[0])     # 4， 256， 28， 28
        sub_fe1 = self.transpose(x_diff[1])    # 4， 512， 14， 14
        sub_fe2 = self.transpose(x_diff[2])    # 4， 1024， 7， 7
        sub_fe3 = self.transpose(x_diff[3])    # 4， 1024， 7， 7
        # 对sub_fe3上采样一次，方便指导sub_fe1
        sub_fe_upsample1 = self.upsample1(sub_fe3)    # 1024 --> 512
        sub_fe_upsample2 = self.upsample2(sub_fe_upsample1)    # 512 --> 256

        # 只需要获取两个交叉注意力即可，即sub_fe3分别与sub_fe2和sub_fe1
        _, _, dgs32 = self.cross2(sub_fe3, sub_fe2)     # 4, 1024, 7, 7
        _, _, dgs31 = self.cross1(sub_fe_upsample1, sub_fe1)   # 4, 512, 14, 14
        _, _, dgs30 = self.cross3(sub_fe_upsample2, sub_fe0)   # 4, 256, 28, 28
        cat32_1 = self.downsample3(self.transpose(x_downsample1[2]))
        cat32_2 = self.downsample3(self.transpose(x_downsample2[2]))

        cat21_1 = self.downsample2(self.transpose(x_downsample1[1]))
        cat21_2 = self.downsample2(self.transpose(x_downsample2[1]))
        cat10_1 = self.downsample1(self.transpose(x_downsample1[0]))
        cat10_2 = self.downsample1(self.transpose(x_downsample2[0]))

        dgs32 = self.upsample1(dgs32)     # 4, 512, 14, 14
        dgs31 = self.upsample2(dgs31)     # 4, 256, 28, 28
        dgs30 = self.upsample3(dgs30)     # 4, 128, 56, 56

        x_downsample1[3] = self.transpose(x_downsample1[3])
        x_downsample2[3] = self.transpose(x_downsample2[3])
        x_downsample1[2] = self.transpose(x_downsample1[2])
        x_downsample2[2] = self.transpose(x_downsample2[2])
        x_downsample1[1] = self.transpose(x_downsample1[1])
        x_downsample2[1] = self.transpose(x_downsample2[1])
        x_downsample1[0] = self.transpose(x_downsample1[0])
        x_downsample2[0] = self.transpose(x_downsample2[0])

        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # 直接将得到的三个特征沿着通道进行cat，然后用2d卷积获取全局信息
        output1 = self.conv1(torch.cat((sub_fe3, x_downsample1[3] + x_downsample2[3], cat32_1 + cat32_2), dim=1)) + sub_fe3
        output2 = self.conv2(torch.cat((dgs32, x_downsample1[2] + x_downsample2[2], cat21_1 + cat21_2), dim=1)) + dgs32
        output3 = self.conv3(torch.cat((dgs31, x_downsample1[1] + x_downsample2[1], cat10_1 + cat10_2), dim=1)) + dgs31
        output4 = self.conv4(torch.cat((dgs30, x_downsample1[0] + x_downsample2[0]), dim=1)) + dgs30
        return output1, output2, output3, output4