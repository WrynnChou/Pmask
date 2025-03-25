import numpy as np
import torch, random
from torchvision import models
with open('utils/conf.json') as f:
    conf = f

model = models.vgg19(pretrained=True)
rho = 0.75
CompressionRate = 1
up = {}
up_v = {}
k_1 = 100.0
for name, params in model.state_dict().items():
    up[name] = params
    threshold = np.percentile(abs(params), 100 - rho * CompressionRate)
    mask2 = torch.lt(abs(params), threshold)
    u1 = torch.sum(torch.square(params[~mask2]))
    u2 = torch.sum(torch.mul(up[name], torch.masked_fill(params, mask2, 0)))
    k_1 = min(k_1, ((u1 + 2 * u2) / (u1 + u2)))
    up_v[name] = (torch.mul(up[name], torch.masked_fill(params, mask2, 0))).sum()
print('fk')