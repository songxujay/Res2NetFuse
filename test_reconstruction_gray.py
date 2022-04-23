# 测试灰度图像重构结果

from net_gray import *

import utils
import cv2
import torch

def main():

    with torch.no_grad():

     for m in range(1,21):

        input_image_dir = './images/test_reconstruction_images/COCO_' + str(m) + '.jpg'
        channel = 1
        [input_image, Height, Width] = utils.read_image(input_image_dir)
        input_image = input_image.reshape(1, 1, input_image.shape[0], input_image.shape[1])
        i = 0
        # model_dir = './gray/batch_4_lr_1e-4_80000_gray/model_gray/scale_'+ str(i) +'_final_model.model'
        model_dir = './gray/model_gray/scale_'+ str(i) +'_final_model.model'
        generator = GenerativeNet(1,1)
        generator.load_state_dict(torch.load(model_dir))
        # params = list(generator.named_parameters())
        # for parameters in generator.parameters():
        #    print(parameters)

        generator.eval()
        generator.cuda()

        temp_img = input_image
        input = torch.from_numpy(temp_img)
        input = input.float()
        input = input.cuda()

        input_ = input
        out_image = generator(input_)
        result = (out_image - torch.min(out_image)) / (torch.max(out_image) - torch.min(out_image) + args.EPSILON)
        result = result * 255

        temp_generative_image = result.cpu()
        temp_generative_image = temp_generative_image.numpy()  # ndarray float32
        temp_generative_image = temp_generative_image.astype(np.uint8)
        temp_generative_image = torch.from_numpy(temp_generative_image)  # tensor
        temp = temp_generative_image.view(temp_generative_image.shape[2], temp_generative_image.shape[3], -1)
        temp = temp.squeeze()
        temp = temp.numpy()
        # save_path = './gray/Res2NetFuse_reconstruction_80000/' + str(m) + '_res2netfuse_80000.jpg'
        save_path = './gray/Res2NetFuse_reconstruction/' + str(m) + '_res2netfuse_a_image.jpg'
        cv2.imwrite(save_path, temp)

if __name__ == '__main__':
    main()