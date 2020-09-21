if __name__ == '__main__':
    import os
    import torch
    import numpy as np
    from options import TestOption
    from pipeline import CustomDataset
    from networks import Generator
    from utils import Manager
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from astropy.io import fits

    torch.backends.cudnn.benchmark = True
    
    opt = TestOption().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    device = torch.device('cuda:0')

    dataset = CustomDataset(opt)
    test_data_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)
    iters = opt.iteration
    step = opt.save_freq
    
    #####################################################################################
    Max_iter = 400000 ######### You can change the Maximum iteration value. #############
    #####################################################################################

    if iters == False :
        for i in range(step,Max_iter+step,step):
            
            ITERATION = int(i)
            path_model = './checkpoints/{}/Model/{}_G.pt'.format(str(STD), str(ITERATION))
            dir_image_save = './checkpoints/{}/Image/Test/{}'.format(str(STD), str(ITERATION))
            os.makedirs(dir_image_save, exist_ok=True)
        
            G = Generator(opt).to(device)
            G.load_state_dict(torch.load(path_model))
            
            manager = Manager(opt)
            
            with torch.no_grad():
                G.eval()
                for input,  name in tqdm(test_data_loader):
                    input = input.to(device)
                    fake = G(input)
                    
                    UpIB = opt.saturation_upper_limit_target
                    LoIB = opt.saturation_lower_limit_target
                        
                    np_fake = fake.cpu().numpy().squeeze() *((UpIB - LoIB)/2) +(UpIB+ LoIB)
                    
                    if opt.data_format_input in ["fits", "fts"]:       
                        fits.writeto(os.path.join(dir_image_save, name[0] + '_AI.fits'), np_fake)
                    elif opt.data_format_input in ["npy"]:
                        np.save(os.path.join(dir_image_save, name[0] + '_AI.fits'), np_fake, allow_pickle=True)
                    else:
                        NotImplementedError("Please check data_format_target option. It has to be fits or npy.")

    else:
        ITERATION = int(iters)
        path_model = './checkpoints/{}/Model/{}_G.pt'.format(str(STD), str(ITERATION))
        dir_image_save = './checkpoints/{}/Image/Test/{}'.format(str(STD), str(ITERATION))
        os.makedirs(dir_image_save, exist_ok=True)
    
        G = Generator(opt).to(device)
        G.load_state_dict(torch.load(path_model))
        
        manager = Manager(opt)
        
        with torch.no_grad():
            G.eval()
            for input,  name in tqdm(test_data_loader):
                input = input.to(device)
                fake = G(input)
                
                UpIB = opt.saturation_upper_limit_target
                LoIB = opt.saturation_lower_limit_target
                    
                np_fake = fake.cpu().numpy().squeeze() *((UpIB - LoIB)/2) +(UpIB+ LoIB)
                
                if opt.data_format_input in ["fits", "fts"]:       
                    fits.writeto(os.path.join(dir_image_save, name[0] + '_AI.fits'), np_fake)
                elif opt.data_format_input in ["npy"]:
                    np.save(os.path.join(dir_image_save, name[0] + '_AI.fits'), np_fake, allow_pickle=True)
                else:
                    NotImplementedError("Please check data_format_target option. It has to be fits or npy.")
