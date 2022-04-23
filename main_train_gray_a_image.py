# 使用单一灰度图像训练网络的主函数
from args_fusion import args
from net_gray import GenerativeNet
from torch.optim import Adam
from tqdm import trange
from scipy.misc import imresize

import matplotlib.pyplot as plt
import utils
import numpy as np
import torch
import pytorch_msssim
import scipy.io as scio
import os
import cv2
import math
import torchvision.models as models

def main():
    original_img_path = './images/train_gray/COCO_train2014_000000000086.jpg'
    train(original_img_path)

def train(original_imgs_path):

    original_image = utils.get_image(original_imgs_path)
    scale_num = 0
    real_images = utils.creat_real_image_pyramid(original_image)

    input_nc = 1
    output_nc = 1

    while scale_num < args.total_number_of_scales:

        if args.resume is not None:
            print('Resuming, initializing using weight from {}.'.format(args.resume))

        input1 = real_images[scale_num] #tensor
        input1 = torch.from_numpy(input1)
        input1 = input1.float()
        train_a_single_scale_network(scale_num, input1, input_nc, output_nc)

        scale_num+=1

    print('\nDone!')


def train_a_single_scale_network(scale_num, input1, input_nc, output_nc):

    generator = GenerativeNet(input_nc,output_nc)
    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        generator.load_state_dict(torch.load(args.resume))

    optimizer = Adam(generator.parameters(), args.lr)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[1000], gamma=args.gamma)
    mse_loss = torch.nn.MSELoss()
    ssim_loss = pytorch_msssim.msssim

    if args.cuda:
        input1 = input1.cuda()
        generator.cuda()

    bar = range(args.epoch_gray[scale_num])
    Loss_pixel = []
    Loss_ssim = []
    Loss_all = []
    height = 256
    width = 256

    print('Start training......')
    for e in bar:

        noise = utils.generate_noise([input_nc, height, width])
        if args.cuda:
            noise = noise.cuda()
        # input = input1 + noise
        input = input1
        real_scale_image = input

        optimizer.zero_grad()
        generative_image = generator(input)# tensor 都是小数值

        if args.cuda:
            real_scale_image =  real_scale_image.cuda()

        pixel_loss_temp = mse_loss(generative_image, real_scale_image)/(generative_image.shape[1] *
                                                generative_image.shape[2] *generative_image.shape[3])
        ssim_loss_temp = ssim_loss(generative_image, real_scale_image, normalize=True)
        ssim_loss_value = (1-ssim_loss_temp)
        pixel_loss_value = pixel_loss_temp

        #total loss
        total_loss = pixel_loss_value + ssim_loss_value
        total_loss.backward(retain_graph=True)#误差反传播
        optimizer.step()#更新参数

        mi = 10
        if(e % mi == 0):
            print("Scale:%d......Epoch:%d......" % (scale_num , e + 1), end=' ')
            print('ssim_loss:%f......pixel_loss:%f......total_loss:%f' % (
            ssim_loss_value, pixel_loss_value, total_loss))
        if(e % mi == 0 ): ###
            Loss_pixel.append(pixel_loss_value.item())
            Loss_ssim.append(ssim_loss_value.item())
            Loss_all.append(total_loss.item())

        #scheduler.step()

    loss_pixel_path = args.save_loss_gray_dir + 'scale_' + str(scale_num) + '_final_pixel.mat'
    scio.savemat(loss_pixel_path,{'final_pixel_loss':Loss_pixel})
    loss_ssim_path = args.save_loss_gray_dir + 'scale_' + str(scale_num) + '_final_ssim.mat'
    scio.savemat(loss_ssim_path, {'final_ssim_loss':Loss_ssim})
    loss_all_path = args.save_loss_gray_dir + 'scale_' + str(scale_num) + '_all_loss.mat'
    scio.savemat(loss_all_path,{'final_all_loss':Loss_all})

    save_model_path = args.save_model_gray_dir + 'scale_' + str(scale_num) + '_final_model.model'
    torch.save(generator.state_dict(), save_model_path)


if __name__ == '__main__':
    main()