import os
import argparse


class BaseOption(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--gpu_ids', type=int, default=0, help='gpu number. If -1, use cpu')
        self.parser.add_argument('--data_format_input', type=str, default='fits',
                                 help="Input data extension. This will be used for loading and saving. [fits or npy]")
        self.parser.add_argument('--data_format_target', type=str, default='fits',
                                 help="Target data extension. This will be used for loading and saving. [fits or npy]")

        # data option
        self.parser.add_argument('--input_ch', type=int, default=3, help="# of input channels for Generater")
        
        self.parser.add_argument('--saturation_lower_limit_input', type=int, default=1, help="Saturation value (lower limit) of input")
        self.parser.add_argument('--saturation_upper_limit_input', type=int, default=200, help="Saturation value (upper limit) of input")
        self.parser.add_argument('--saturation_lower_limit_target', type=int, default=-3000, help="Saturation value (lower limit) of target")
        self.parser.add_argument('--saturation_upper_limit_target', type=int, default=3000, help="Saturation value (upper limit) of target")

        # data augmentation
        self.parser.add_argument('--batch_size', type=int, default=1, help='the number of batch_size')
        self.parser.add_argument('--dataset_name', type=str, default='AIA_to_HMI/', help='[dataset directory name')
        self.parser.add_argument('--data_type', type=int, default=32, help='float dtype')
        self.parser.add_argument('--image_mode', type=str, default='png', help='extension for saving image')
        self.parser.add_argument('--n_downsample', type=int, default=5, help='how many times you want to downsample input data in G')
        self.parser.add_argument('--n_residual', type=int, default=9, help='the number of residual blocks in G')
        self.parser.add_argument('--n_workers', type=int, default=1, help='how many threads you want to use')
        self.parser.add_argument('--norm_type', type=str, default='InstanceNorm2d', help='[BatchNorm2d, InstanceNorm2d]')
        self.parser.add_argument('--val_during_train', action='store_true', default=False)

    def parse(self):
        opt = self.parser.parse_args()
        opt.format = 'png'
        opt.n_df = 64
        opt.flip = False

        opt.n_gf = 32
        opt.output_ch = 1

        if opt.data_type == 16:
            opt.eps = 1e-4
        elif opt.data_type == 32:
            opt.eps = 1e-8

        dataset_name = opt.dataset_name
        model_name = "pix2pixHD"

        os.makedirs(os.path.join('./checkpoints', dataset_name, 'Image', 'Train', model_name), exist_ok=True)
        os.makedirs(os.path.join('./checkpoints', dataset_name, 'Image', 'Test', model_name), exist_ok=True)
        os.makedirs(os.path.join('./checkpoints', dataset_name, 'Model', model_name), exist_ok=True)

        if opt.is_train:
            opt.image_dir = os.path.join('./checkpoints', dataset_name, 'Image/Train', model_name)
        else:
            opt.image_dir = os.path.join('./checkpoints', dataset_name, 'Image/Test', model_name)

        opt.model_dir = os.path.join('./checkpoints', dataset_name, 'Model', model_name)
        
        
        return opt


class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        self.parser.add_argument('--is_train', type=bool, default=True, help='train flag')
        self.parser.add_argument('--n_epochs', type=int, default=150, help='how many epochs you want to train')
        self.parser.add_argument('--latest', type=int, default=0, help='Resume epoch')

        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--epoch_decay', type=int, default=100, help='when to start decay the lr')
        self.parser.add_argument('--GAN_type', type=str, default='LSGAN', help='[GAN, LSGAN, WGAN_GP]')
        self.parser.add_argument('--lambda_FM', type=int, default=10, help='weight for FM loss')
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--n_D', type=int, default=2, help='how many discriminators in differet scales you want to use')
        
        self.parser.add_argument('--report_freq', type=int, default=10)
        self.parser.add_argument('--save_freq', type=int, default=5000)
        self.parser.add_argument('--display_freq', type=int, default=100)
        self.parser.add_argument('--no_shuffle', action='store_true', default=False, help='if you want to shuffle the order')
        


class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()
        
        self.parser.add_argument('--is_train', type=bool, default=False, help='test flag')
        self.parser.add_argument('--iteration', type=bool, default=False, help='if you want to generate from input for the specific iteration')
        self.parser.add_argument('--no_shuffle', type=bool, default=True, help='if you want to shuffle the order')
