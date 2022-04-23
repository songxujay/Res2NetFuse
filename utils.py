import torch
import torch.nn as nn
import numpy as np
import random

from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from scipy.misc import  imread, imsave, imresize
from args_fusion import args

def get_train_images_auto(images):
    outputs = []
    for image in images:
        image = np.reshape(image, [image.shape[2], image.shape[3], image.shape[1]])
        outputs.append(image)

    outputs = np.stack(outputs, axis=0)
    outputs = torch.from_numpy(outputs).float()
    outputs = outputs.transpose(1,3)
    outputs = outputs.transpose(2,3)
    return outputs

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images

#用来存储彩色红外可见光的融合结果
def save_images(paths, datas, save_path, prefix=None, suffix=None):
    if isinstance(paths, str):
        paths = [paths]

    t1 = len(paths)
    t2 = len(datas)
    assert (len(paths) == len(datas))

    if not exists(save_path):
        mkdir(save_path)

    if prefix is None:
        prefix = ''
    if suffix is None:
        suffix = ''

    for i, path in enumerate(paths):
        data = datas[i]
        # print('data ==>>\n', data)
        # if data.shape[2] == 1:
        #     data = data.reshape([data.shape[0], data.shape[1]])
        # print('data reshape==>>\n', data)
        data = data.squeeze()

        name, ext = splitext(path)
        name = name.split(sep)[-1]

        path = join(save_path, prefix + suffix + ext)
        print('data path==>>', path)

        # new_im = Image.fromarray(data)
        # new_im.show()

        imsave(path, data)

def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        # noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = torch.randn(num_samp, size[0], round(size[1] / scale), round(size[2] / scale))
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

#训练的时候读取图片，训练图片大小为256x256
def get_image(path, height=256, width=256, flag = False):
    if flag is True:
        image = imread(path, mode='RGB')
    else:
        image = imread(path,mode = 'L')

    if height is not  None and width is not  None:
        image = imresize(image, [height, width], interp='nearest')

    return image

#测试的时候读取图片
def read_image(path, flag = False):
    if flag is True:
        image = imread(path, mode='RGB')
    else:
        image = imread(path,mode = 'L')
    shape = image.shape
    temp_height = int(shape[0])
    temp_width = int(shape[1])
    temp_image = imresize(image, (temp_height, temp_width))
    return temp_image, shape[0], shape[1]

def creat_real_image_pyramid(original_image, height=256, width=256):
    real_images = []
    size = args.min_size
    for i in range(0, args.total_number_of_scales, 1):
        temp_height = int(size)
        temp_width = int(size)
        temp_real_image = imresize(original_image, (temp_height, temp_width))
        temp_real_image = np.reshape(temp_real_image, [1, 1, temp_height, temp_width])
        real_images.append(temp_real_image)

    return real_images

def creat_real_image_pyramid_3_channels(original_image, height=256, width=256):
    real_images = []
    size = args.min_size
    for i in range(0, args.total_number_of_scales, 1):
        temp_height = int(size)
        temp_width = int(size)
        temp_real_image = imresize(original_image, (temp_height, temp_width))
        temp_real_image = np.reshape(temp_real_image, [1, 3, temp_height, temp_width])
        real_images.append(temp_real_image)

    return real_images

def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp,0,1)
    return inp

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t