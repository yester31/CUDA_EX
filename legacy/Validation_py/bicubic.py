import torch
import os
import argparse
import time
import numpy as np
from compare import *

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
device = "cpu:0"

def bicubic_torch(input_c, output_c, N, C, H, W):
    print('execute pytorch bicubic interpolation!!!')

    input_torch_tensor = torch.Tensor(input_c).view(N, C, H, W).to(device)

    bicubic_torch = torch.nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True).to(device)

    begin = time.time()

    output_py = bicubic_torch(input_torch_tensor)
    #torch.cuda.synchronize()

    dur = time.time() - begin

    output_py = output_py.detach().cpu().data.numpy().flatten()

    compare_two_tensor(output_py, output_c)
    print('dur time(pytorch) : %.3f [msec]'%(dur * 1000))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='add parameters')

    parser.add_argument('--N', type=int, default=1, help='batch size')
    parser.add_argument('--C', type=int, default=3, help='channel')
    parser.add_argument('--H', type=int, default=1080, help='height')
    parser.add_argument('--W', type=int, default=1920, help='width')

    args = parser.parse_args()

    dir_path = os.path.dirname(__file__)

    N = args.N
    C = args.C
    H = args.H
    W = args.W
    output_c = np.fromfile(os.path.join(dir_path, 'Output_C'), dtype=np.float32)
    input_c = np.fromfile(os.path.join(dir_path, 'Input_C'), dtype=np.float32)

    bicubic_torch(input_c, output_c, N, C, H, W)