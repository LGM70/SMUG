import torch
import torch.nn as nn
import numpy as np
from models.didn import DIDN
from models import networks
from util.metrics import PSNR
import pytorch_msssim
import global_network_dataset2
import matplotlib.pyplot as plt
from options.test_options import TestOptions

opt = TestOptions().parse()

opt.batchSize = 1

device = torch.device(('cuda:' + str(opt.gpu_ids[0])) if torch.cuda.is_available() else 'cpu')

netG = DIDN(2, 2, num_chans=64, pad_data=True, global_residual=True, n_res_blocks=2)
netG.load_state_dict(torch.load(opt.netGpath, map_location=device))
netG = netG.float().to(device)

# Loss and optimizer
mse_loss = nn.MSELoss().to(device)
ssim_loss = pytorch_msssim.SSIM(data_range=2.0, channel=2).to(device)

def CG(output, tol, L, smap, mask, alised_image):
    return networks.CG.apply(output, tol, L, smap, mask, alised_image)

def Recon(cg_iter, smap, mask, input, smoothing, num_sample, epsilon):
    output_CG = input
    if smoothing == 'none': # vanilla MoDL
        for _ in range(cg_iter):
            output_NN = netG(output_CG)
            output_CG = CG(output_NN, tol=opt.CGtol, L=opt.Lambda, smap=smap, mask=mask, alised_image=input)
    elif smoothing == 'RSE2E':
        input_i = input.repeat(num_sample, 1, 1, 1)
        noises = torch.normal(0, epsilon, input_i.shape).to(device)
        noised_input = torch.clamp(noises + input_i, min=-1, max=1)
        output_CG = noised_input
        
        for _ in range(cg_iter):
            output_NN = netG(output_CG)
            output_CG = CG(output_NN, tol=opt.CGtol, L=opt.Lambda, smap=smap.repeat(num_sample, 1, 1, 1, 1), mask=mask.repeat(num_sample, 1, 1, 1), alised_image=input_i)
        
        output_final = torch.zeros_like(input).to(device)
        for j in range(opt.batchSize):
            output_final[j, :, :, :] = torch.sum(output_CG[j::opt.batchSize, :, :, :], 0)
        output_CG = output_final / num_sample
    elif smoothing == 'SMUGv0':
        for _ in range(cg_iter):
            output_CG = output_CG.repeat(num_sample, 1, 1, 1)
            noises = torch.normal(0, epsilon, output_CG.shape).to(device)
            noised_input = torch.clamp(noises + output_CG, min=-1, max=1)

            output_NN = netG(noised_input)
            output_CG = CG(output_NN, tol=opt.CGtol, L=opt.Lambda, smap=smap.repeat(num_sample, 1, 1, 1, 1), mask=mask.repeat(num_sample, 1, 1, 1), alised_image=input.repeat(num_sample, 1, 1, 1))

            output_CG_final = torch.zeros_like(input).to(device)
            for j in range(opt.batchSize):
                output_CG_final[j, :, :, :] = torch.sum(output_CG[j::opt.batchSize, :, :, :], 0)
            output_CG = output_CG_final / num_sample
    elif smoothing == 'SMUG':
        # the same noises are used in every iteration/unrolling step
        noises = torch.normal(0, epsilon, output_CG.repeat(num_sample, 1, 1, 1).shape).to(device)
        for _ in range(cg_iter):
            output_CG = output_CG.repeat(num_sample, 1, 1, 1)
            noised_input = torch.clamp(noises + output_CG, min=-1, max=1)

            output_NN = netG(noised_input)
            output_NN_final = torch.zeros_like(input).to(device)
            for j in range(opt.batchSize):
                output_NN_final[j, :, :, :] = torch.sum(output_NN[j::opt.batchSize, :, :, :], 0)
            output_NN_final /= num_sample

            output_CG = CG(output_NN_final, tol=opt.CGtol, L=opt.Lambda, smap=smap, mask=mask, alised_image=input)

    return output_CG

acceleration, _, test_loader = global_network_dataset2.loadData(opt.dataroot, opt.mask_dataroot, opt.train_valiSize, opt.testSize, opt.batchSize, opt.acceleration)

test_psnr = []
test_ssim = []

for i, (test_direct, test_target, test_smap, test_mask) in enumerate(test_loader):
    vis = opt.visualize and i == 2

    test_input = test_direct.to(device).float()
    test_smap = test_smap.to(device).float()
    test_mask = test_mask.to(device).float()
    test_label = test_target.to(device).float()
    clean_test_input = test_input

    with torch.no_grad():
        test_result = Recon(opt.blockIter, test_smap, test_mask, clean_test_input, opt.smoothing, opt.num_sample, opt.smoothing_epsilon)
        psnr_test = PSNR(test_label, test_result)
        ssim_test = ssim_loss(test_label, test_result)
        test_psnr.append(float(psnr_test))
        test_ssim.append(float(ssim_test))
        
    if vis:
        img_out = test_result.detach().cpu().numpy()
        img_out = img_out.squeeze(0)
        img_out = img_out[0] + img_out[1] * 1j

        plt.imshow(np.abs(img_out), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        img_name = opt.netGpath[:-4] + '_' + str(opt.acceleration) + '_acceleration.pdf'
        plt.savefig(img_name, dpi=600)
        plt.close()
    
message = ''
message += f'acceleration factor: {acceleration:.4f}\n'
message += f'test RSNR: {np.average(test_psnr):.4f} ± {np.std(test_psnr):.4f}\n'
message += f'test SSIM: {np.average(test_ssim):.4f} ± {np.std(test_ssim):.4f}\n'
message += '\n'

file_name = opt.netGpath[:-4] + '_' + str(opt.acceleration) + '_acceleration_test.out'
with open(file_name, 'w') as result_file:
    result_file.write(message)

print(message)