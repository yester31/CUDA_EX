import torch
import os
import argparse
import numpy as np
from compare import *

def bicubic_torch(input_c, output_c, N, C, H, W):
    print('execute pytorch bicubic interpolation!!!')
    input_torch_tensor = torch.Tensor(input_c).view(N, C, H, W)

    bicubic_torch = torch.nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)

    output_py = bicubic_torch(input_torch_tensor)

    output_py = output_py.detach().numpy().flatten()

    compare_two_tensor(output_py, output_c)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='add parameters')

    parser.add_argument('--N', type=int, default=1, help='batch size')
    parser.add_argument('--C', type=int, default=3, help='channel')
    parser.add_argument('--H', type=int, default=4, help='height')
    parser.add_argument('--W', type=int, default=4, help='width')

    args = parser.parse_args()

    dir_path = os.path.dirname(__file__)

    N = args.N
    C = args.C
    H = args.H
    W = args.W
    output_c = np.fromfile(os.path.join(dir_path, 'Output_C'), dtype=np.float32)
    input_c = np.fromfile(os.path.join(dir_path, 'Input_C'), dtype=np.float32)

    bicubic_torch(input_c, output_c, N, C, H, W)