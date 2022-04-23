#融合策略以及用于测试的网络结构
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2

from args_fusion import args

#定义卷积层
class ConvLayer(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, stride, pad, is_last = False):
        '''
        初始化
        :param in_channel:输入特征通道数
        :param out_channel: 输出特征通道数
        :param ker_size: 卷积核大小
        :param stride: 步长
        :param pad: 填充尺寸
        :param is_last: 是否为最后一层
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
        :param in_channels:输入特征通道数
        :param kernel_size:卷积核大小
        :param stride:步长
        :param pad:填充尺寸
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

def fusion(en1, en2, strategy_type = 'addition'):
    '''
    定义融合方法
    :param en1:第一幅待融合图像的特征
    :param en2:第二幅待融合图像的特征
    :param strategy_type:融合策略，默认为基于相加的融合策略
    :return:融合后的特征
    '''
    #相加融合策略
    if strategy_type == 'addition':
        f = en1 + en2

    #基于一范数空间注意力机制的融合策略
    if strategy_type == 'l1-norm':
        en1_ = abs(en1)
        en2_ = abs(en2)
        en1_ = en1_.sum(dim=1)
        en2_ = en2_.sum(dim=1)
        en1_ = torch.squeeze(en1_)
        en2_ = torch.squeeze(en2_)
        # caculate the map for source images
        mask_value = en1_ + en2_

        mask_sign_a = en1_ / mask_value
        mask_sign_b = en2_ / mask_value

        f = mask_sign_a*en1 + mask_sign_b*en2

    #基于平均操作空间注意力机制的融合策略
    if strategy_type == 'mean':
        spatial1 = en1.mean(dim=1, keepdim=True)
        spatial2 = en2.mean(dim=1, keepdim=True)
        spatial1 = (spatial1 - torch.min(spatial1)) / (
                torch.max(spatial1) - torch.min(spatial1) + args.EPSILON)
        spatial2 = (spatial2 - torch.min(spatial2)) / (
                torch.max(spatial2) - torch.min(spatial2) + args.EPSILON)
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + args.EPSILON)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + args.EPSILON)
        f = spatial_w1 * en1 + spatial_w2 * en2

    return f

# generative network
class GenerativeNet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
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


    def forward(self, x1, x2):
        c1 = self.conv(x1)
        c2 = self.conv(x2)
        res1 = self.res2netBlock(c1)
        res2 = self.res2netBlock(c2)

        en1 = c1 + res1
        en2 = c2 + res2

        #mean，l1-norm
        f = fusion(en1, en2, 'l1-norm')

        result = self.decoder(f)

        return result