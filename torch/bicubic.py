import torch
import numpy as np


a = np.array([10, 20, 30, 40])
b = torch.Tensor(a).view(1, 1, 2, 2)

m = torch.nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
print(m(b))
