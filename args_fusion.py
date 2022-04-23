#网络参数设置
class args():

    cuda = 1  # set 1 for running on GPU, 0 for CPU
    #device = 'cuda:0'
    epoch_gray = [2000]#单张灰度图像训练模型迭代次数

    batch_size  = 4#80000张灰度图像训练模型的批量大小

    save_model_gray_dir = './gray/model_gray/'
    save_loss_gray_dir = './gray/loss_gray/'
    save_generative_images_gray_dir = './gray/generative_images_gray/'
    save_gray_IR_VIS_results_dir = './gray/fusion_results_gray/'
    save_recons_result_gray_dir = './gray/reconstruction_results_gray/'

    save_model_color_dir = './color/model_color/'
    save_loss_color_dir = './color/loss_color/'
    save_generative_images_color_dir = './color/generative_images_color/'
    save_color_IR_VIS_dir = './color/fusion_results_color/'
    save_recons_result_color_dir = './color/reconstruction_results_color/'

    save_MF_results_dir = './MF_results/'

    #train
    total_number_of_scales = 1
    min_size = 256
    gamma = 0.1

    lr = 5e-4 #learning rate, default is 0.0001
    EPSILON = 1e-5

    resume = None

    #network parameters
    nb_filter = [16, 32, 64]  # nb是中间卷积层固定的通道数
    kernel_size = 3
    stride = 1
    pad = 1
