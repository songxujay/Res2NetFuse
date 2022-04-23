# 测试彩色图像重构结果

from net_gray import *

import utils
import cv2
import torch
import matplotlib.pyplot as plt

def main():

    with torch.no_grad():

     for m in range(1,6):

        input_image_dir = './images/test_images_color/' + str(m) + '.jpg'
        input_image = cv2.imread(input_image_dir)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YCrCb)
        Height = input_image.shape[0]
        Width = input_image.shape[1]
        Y, Cr, Cb = cv2.split(input_image)
        input_Y = Y.reshape(1, 1, Y.shape[0], Y.shape[1])
        i = 0
        model_dir = './gray/model_gray/scale_'+ str(i) +'_final_model.model'
        generator = GenerativeNet(1,1)
        generator.load_state_dict(torch.load(model_dir))
        generator.eval()
        generator.cuda()

        temp_img = input_Y
        input = torch.from_numpy(temp_img)
        input = input.float()
        input = input.cuda()

        input_ = input
        out_image = generator(input_)
        result = out_image
        # result = (out_image - torch.min(out_image)) / (torch.max(out_image) - torch.min(out_image) + args.EPSILON)
        # result = result * 255

        temp_generative_image_Y = result.cpu()
        temp_generative_image_Y = temp_generative_image_Y.numpy()  # ndarray float32
        temp_generative_image_Y = temp_generative_image_Y.astype(np.uint8)
        temp_generative_image_Y = torch.from_numpy(temp_generative_image_Y)  # tensor
        temp_generative_image_Y = temp_generative_image_Y.view(Height, Width, -1)

        temp_generative_image_Cr = Cr
        temp_generative_image_Cr = torch.from_numpy(temp_generative_image_Cr)  # tensor
        temp_generative_image_Cr = temp_generative_image_Cr.view(Height, Width, -1)

        temp_generative_image_Cb = Cb
        temp_generative_image_Cb = torch.from_numpy(temp_generative_image_Cb)  # tensor
        temp_generative_image_Cb = temp_generative_image_Cb.view(Height, Width, -1)

        outputY = temp_generative_image_Y.reshape([1, Height, Width])#Tensor
        outputCr = temp_generative_image_Cr.reshape([1, Height, Width])#Tensor
        outputCb = temp_generative_image_Cb.reshape([1, Height, Width])#Tensor

        output = np.stack([outputY, outputCr, outputCb], axis=-1)

        output = output.squeeze()
        output_path = args.save_recons_result_color_dir + str(m) + '.png'
        result = cv2.cvtColor(output, cv2.COLOR_YCrCb2RGB)
        plt.imsave(output_path, result)

if __name__ == '__main__':
    main()