#融合RGB和红外图像
from net_fusion_gray import GenerativeNet
from args_fusion import  args

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

for k in range(1,5):
    print(k)
    input_image_dir1 = './images/color_IR_VIS/IR_'+ str(k) +'.jpg'
    input_image_dir2 = './images/color_IR_VIS/VIS_'+ str(k) +'.jpg'
    channel = 1

    input_image1 = cv2.imread(input_image_dir1)
    input_image1 = cv2.cvtColor(input_image1, cv2.COLOR_BGR2GRAY)
    # output_path = args.save_color_IR_VIS_dir + str(k) + '.png'
    # cv2.imwrite(output_path, input_image1)
    input_image2 = cv2.imread(input_image_dir2)
    input_image2 = cv2.cvtColor(input_image2, cv2.COLOR_BGR2YCrCb)
    Y2, Cr2, Cb2 = cv2.split(input_image2)
    Height = input_image1.shape[0]
    Width = input_image1.shape[1]

    input_image1_Y = input_image1.reshape(1, 1, Height,Width)
    input_image2_Y = Y2.reshape(1, 1, Height,Width)

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


            temp_img1_Y = input_image1_Y
            temp_img2_Y = input_image2_Y

            input1_Y = torch.from_numpy(temp_img1_Y)
            input2_Y = torch.from_numpy(temp_img2_Y)


            input_1_Y = input1_Y.float().cuda()
            input_2_Y = input2_Y.float().cuda()

            out_image_Y = generator(input_1_Y, input_2_Y)

        result_Y = (out_image_Y - torch.min(out_image_Y)) / (torch.max(out_image_Y) - torch.min(out_image_Y) + args.EPSILON)
        result_Y = result_Y * 255
        result_Cr = Cr2
        result_Cb = Cb2

        temp_generative_image_Y = result_Y.cpu()
        temp_generative_image_Y = temp_generative_image_Y.numpy()  # ndarray float32
        temp_generative_image_Y = temp_generative_image_Y.astype(np.uint8)
        temp_generative_image_Y = torch.from_numpy(temp_generative_image_Y)  # tensor
        temp_generative_image_Y = temp_generative_image_Y.view(Height, Width, -1)

        temp_generative_image_Cr = result_Cr
        temp_generative_image_Cr = torch.from_numpy(temp_generative_image_Cr)  # tensor
        temp_generative_image_Cr = temp_generative_image_Cr.view(Height, Width, -1)

        temp_generative_image_Cb = result_Cb
        temp_generative_image_Cb = torch.from_numpy(temp_generative_image_Cb)  # tensor
        temp_generative_image_Cb = temp_generative_image_Cb.view(Height, Width, -1)

        outputY = temp_generative_image_Y.reshape([1, Height, Width])#Tensor
        outputCr = temp_generative_image_Cr.reshape([1, Height, Width])#Tensor
        outputCb = temp_generative_image_Cb.reshape([1, Height, Width])#Tensor

        output = np.stack([outputY, outputCr, outputCb], axis=-1)

        output = output.squeeze()
        output_path = args.save_color_IR_VIS_dir + str(k) + '.png'
        result = cv2.cvtColor(output, cv2.COLOR_YCrCb2RGB)
        plt.imsave(output_path, result)

