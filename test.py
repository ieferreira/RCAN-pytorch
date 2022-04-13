import argparse
import os
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from model import RCAN
from math import log10, sqrt
import cv2
import numpy as np

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def PSNR(original, compressed):
    """Calculates PSNR between two images"""
    original = np.array(original)
    compressed = np.array(compressed)
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='RCAN')
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    model = RCAN(opt)

    state_dict = model.state_dict()
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model = model.to(device)
    model.eval()
    
    psnr_bicubic = []
    psnr_rcan = []
    

    for i in os.listdir(opt.images_dir):
        filename = os.path.basename(opt.images_dir+"/"+i).split('.')[0]

        input = pil_image.open(opt.images_dir+"/"+i).convert('RGB')
        orig = pil_image.open(opt.images_dir+"/"+i).convert('RGB')
        
        lr = input.resize((input.width // opt.scale, input.height // opt.scale), pil_image.BICUBIC)

        bicubic = lr.resize((input.width, input.height), pil_image.BICUBIC)
        bicubic.save(os.path.join(opt.outputs_dir, '{}_x{}_bicubic.png'.format(filename, opt.scale)))

        input = transforms.ToTensor()(lr).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(input)

        output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        output = pil_image.fromarray(output, mode='RGB')
        
        psnr_bicubic_image = PSNR(orig, bicubic)
        psnr_rcan_image = PSNR(orig, output) 
        
        print(f"PSNR image {i} bicubic: {psnr_bicubic_image}")
        print(f"PSNR image {i} RCAN: {psnr_rcan_image}")
        psnr_bicubic.append(psnr_bicubic_image)
        psnr_rcan.append(psnr_rcan_image)
        
        output.save(os.path.join(opt.outputs_dir, '{}_x{}_{}.png'.format(filename, opt.scale, opt.arch)))

print(f"Average bicubic PSNR: {sum(psnr_bicubic)/len(psnr_bicubic)}")
print(f"Average RCAN PSNR: {sum(psnr_rcan)/len(psnr_rcan)}")