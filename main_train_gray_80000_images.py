#使用80000张灰度图像训练网络的主函数
from args_fusion import args
from net_gray import GenerativeNet
from torch.optim import Adam
from tqdm import trange

import matplotlib.pyplot as plt
import utils
import numpy as np
import torch
import pytorch_msssim
import scipy.io as scio
import cv2
import random


def main():
    train()

def train():
    original_imgs_path = utils.list_images('D:\database\\COCO\\train\\')
    num_train_images = len(original_imgs_path)
    original_images = []
    real_images = []
    for i in range(num_train_images):
        temp_original = cv2.imread(original_imgs_path[i],)
        temp_original = cv2.resize(temp_original, (256, 256))
        temp_original = cv2.cvtColor(temp_original, cv2.COLOR_BGR2GRAY)
        temp_original = np.reshape(temp_original, [1, 1, 256, 256])
        real_images.append(temp_original)

    scale_num = 0
    while scale_num < args.total_number_of_scales:

        if args.resume is not None:
            print('Resuming, initializing using weight from {}.'.format(args.resume))
        train_a_single_scale_network(scale_num, real_images, input_nc=1, output_nc=1)

        scale_num += 1

    print('\nDone!')


def train_a_single_scale_network(scale_num, real_images, input_nc, output_nc):
    generator = GenerativeNet(input_nc, output_nc)
    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        generator.load_state_dict(torch.load(args.resume))

    lr = 1e-4
    optimizer = Adam(generator.parameters(), lr)
    mse_loss = torch.nn.MSELoss()
    ssim_loss = pytorch_msssim.msssim

    if args.cuda:
        generator.cuda()

    bar = range(args.epoch_color[scale_num])
    Loss_pixel = []
    Loss_ssim = []
    Loss_all = []
    num_train_images = len(real_images)
    batch_size = args.batch_size
    mod = num_train_images % batch_size
    print('BATCH SIZE %d.' % batch_size)
    print('Train images number %d.' % num_train_images)
    print('Train images samples %s.' % str(num_train_images / batch_size))
    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        real_images = real_images[:-mod]
    batches = int(len(real_images) // batch_size)
    step = 0
    print('Start training......')
    for e in bar:
        random.shuffle(real_images)
        for batch in range(batches):
            step += 1
            inputs = real_images[batch * batch_size:(batch * batch_size + batch_size)]  # tensor
            input1 = utils.get_train_images_auto(inputs)
            if args.cuda:
                input1 = input1.cuda()
            input = input1
            real_scale_image = input
            optimizer.zero_grad()
            generative_image = generator(input)
            if args.cuda:
                real_scale_image = real_scale_image.cuda()

            pixel_loss_temp = mse_loss(generative_image, real_scale_image) / (batch_size * generative_image.shape[1] *
                                                                              generative_image.shape[2] *
                                                                              generative_image.shape[3])
            ssim_loss_temp = ssim_loss(generative_image, real_scale_image, normalize=True)
            ssim_loss_value = (1 - ssim_loss_temp)
            pixel_loss_value = pixel_loss_temp
            # total loss
            total_loss = pixel_loss_value + ssim_loss_value
            total_loss.backward(retain_graph=True)
            optimizer.step()

            mi = 2000

            if (step % mi == 0):
                print("Epoch:%d......step:%d......" % (e, step), end=' ')
                print('ssim_loss:%f......pixel_loss:%f......total_loss:%f' % (
                    ssim_loss_value, pixel_loss_value, total_loss))

            if (step % mi == 0):  ###
                Loss_pixel.append(pixel_loss_value.item())
                Loss_ssim.append(ssim_loss_value.item())
                Loss_all.append(total_loss.item())

        # scheduler.step()

    loss_pixel_path = args.save_loss_gray_dir + 'scale_' + str(scale_num) + '_final_pixel.mat'
    scio.savemat(loss_pixel_path, {'final_pixel_loss': Loss_pixel})
    loss_ssim_path = args.save_loss_gray_dir + 'scale_' + str(scale_num) + '_final_ssim.mat'
    scio.savemat(loss_ssim_path, {'final_ssim_loss': Loss_ssim})
    loss_all_path = args.save_loss_gray_dir + 'scale_' + str(scale_num) + '_all_loss.mat'
    scio.savemat(loss_all_path, {'final_all_loss': Loss_all})

    save_model_path = args.save_model_gray_dir + 'scale_' + str(scale_num) + '_final_model.model'
    torch.save(generator.state_dict(), save_model_path)


if __name__ == '__main__':
    main()