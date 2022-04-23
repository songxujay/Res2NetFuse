#使用灰度图像训练的网络结构
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from args_fusion import args

#定义卷积层
class ConvLayer(nn.Sequential):

    def __init__(self, in_channel, out_channel, ker_size, stride, pad, is_last = False):
        '''
        初始化
        :param in_channel: 输入特征通道数
        :param out_channel:输出特征通道数
        :param ker_size:卷积核大小
        :param stride:卷积步长
        :param pad:填充大小
        :param is_last:是否为最后一层
        '''
        super(ConvLayer , self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, ker_size, stride, pad)
        self.is_last = is_last

    def forward(self, x):
        x = self.conv(x)
        if self.is_last is False:
            x = F.relu(x, inplace=True)
        return x

#定义Res2Net模块
class Res2NetBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride, pad):
        '''
        初始化
        :param in_channels: 输入特征通道数
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param pad: 填充尺寸
        '''
        super(Res2NetBlock, self).__init__()
        self.stage = 4
        out_channels_def = int(in_channels / self.stage)
        self.width = out_channels_def
        self.conv = ConvLayer(out_channels_def, out_channels_def, kernel_size, stride, pad)

    def forward(self, x):
        spx = torch.split(x, self.width, 1)
        for i in range(0, self.stage):
            if i == 0:
                x = spx[i]
            elif i == 1:
                x = spx[i]
            else:
                x = y + spx[i]

            if i==0:
                out = x
            else:
                y = self.conv(x)
                y = self.conv(y)
                out = torch.cat((out, y), 1)


        return out

# generative network
class GenerativeNet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        '''
        初始化
        :param input_nc:输入图像通道数
        :param output_nc: 输出图像通道数
        '''
        super(GenerativeNet, self).__init__()
        self.is_cuda = torch.cuda.is_available()

        res2net = Res2NetBlock
        nb_filter = args.nb_filter #nb是中间卷积层固定的通道数
        kernel_size = args.kernel_size
        stride = args.stride
        pad = args.pad

        #  encoder
        self.conv = nn.Sequential(
            ConvLayer(input_nc, nb_filter[1], kernel_size, stride, pad),  # 1x32
            ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride, pad)  # 32x64
        )
        self.res2netBlock = nn.Sequential(
            ConvLayer(nb_filter[2], nb_filter[2], 1, stride, 0),  # 64x64  kernel_size=1
            res2net(nb_filter[2], kernel_size, stride, pad),
            ConvLayer(nb_filter[2], nb_filter[2], 1, stride, 0)  # 64x64 kernel_size=1
        )

        # decoder
        self.decoder = nn.Sequential(
            ConvLayer(nb_filter[0] * 4, nb_filter[2], kernel_size, stride, pad),  # 64 x 64
            ConvLayer(nb_filter[2], nb_filter[1], kernel_size, stride, pad), #64 x 32
            ConvLayer(nb_filter[1], nb_filter[0], kernel_size, stride, pad), #32 x 16
            ConvLayer(nb_filter[0], output_nc, kernel_size, stride, pad, True) #16 x 1
        )


    def forward(self, x):
        c = self.conv(x)
        res = self.res2netBlock(c)
        result = self.decoder(c + res)

        return result