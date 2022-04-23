#使用单一自然图像训练的网络融合灰度红外可见光图像
from net_fusion_gray import GenerativeNet
from args_fusion import  args

import numpy as np
import utils
import cv2
import torch

for k in range(1,21):#1,21
    print(k)
    input_image_dir1 = './images/infrared_visible/IR'+ str(k) +'.png'
    input_image_dir2 = './images/infrared_visible/VIS' + str(k) + '.png'
    channel = 1

    [input_image1, Height, Width] = utils.read_image(input_image_dir1)
    [input_image2, Height, Width]  = utils.read_image(input_image_dir2)

    input_image1 = input_image1.reshape(1, 1, input_image1.shape[0], input_image1.shape[1])
    input_image2 = input_image2.reshape(1, 1, input_image2.shape[0], input_image2.shape[1])

    with torch.no_grad():

        begin = 0
        last = 1
        for i in range(begin, last):
            model_dir = './gray/model_gray/scale_'+ str(i) +'_final_model.model'
            generator = GenerativeNet(1,1)
            generator.load_state_dict(torch.load(model_dir))
            generator.eval()
            if torch.cuda.is_available():
                generator.cuda()

            temp_img1 = input_image1
            temp_img2 = input_image2

            real_1 = torch.from_numpy(temp_img1)
            real_1 = real_1.float()
            input1 = utils.upsampling(real_1, Height, Width)  # tensor
            if torch.cuda.is_available():
                input1 = input1.cuda()

            real_2 = torch.from_numpy(temp_img2)
            real_2 = real_2.float()
            input2 = utils.upsampling(real_2, Height, Width)  # tensor
            if torch.cuda.is_available():
                input2 = input2.cuda()

            input_1 = input1
            input_2 = input2
            out_image = generator(input_1, input_2)

        result = (out_image - torch.min(out_image)) / (torch.max(out_image) - torch.min(out_image) + args.EPSILON)
        result = result * 255

        temp_generative_image = result.cpu()
        temp_generative_image = temp_generative_image.numpy()  # ndarray float32
        temp_generative_image = temp_generative_image.astype(np.uint8)
        temp_generative_image = torch.from_numpy(temp_generative_image)  # tensor
        temp = temp_generative_image.view(temp_generative_image.shape[2], temp_generative_image.shape[3], -1)
        temp = temp.squeeze()
        temp = temp.numpy()
        result_index = 10 + k
        cv2.imwrite('%s%d_l1-norm.png' % ('./gray/res2net_fusion_results_gray_l1-norm/', result_index), temp)

