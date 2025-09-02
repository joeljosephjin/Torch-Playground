import os
import random

import numpy as np
import torch
import torch.nn as nn

from data.data import load_cifar_10, load_cifar_10_other
from utils import set_seed

# seed=42

# np.random.seed(seed)
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# # When running on the CuDNN backend, two further options must be set
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# # Set a fixed value for the hash seed
# os.environ["PYTHONHASHSEED"] = str(seed)
# print(f"Random seed set as {seed}")

set_seed()


trainldr, validldr = load_cifar_10()
# import pdb; pdb.set_trace()
# x1 = next(iter(trainldr))
# print(x1[-1][-1], 'id', id(trainldr))
# print(x1[-1][-1], 'id', id(x1))
# for x in trainldr:
#     print(x[-1][-1])
#     break

# trainldr, validldr = load_cifar_10()
# for x in trainldr:
#     print(x[-1][-1])
#     break


# trainldr_other, validldr_other = load_cifar_10_other()


x1 = list(trainldr)
# x1 = list(trainldr_other)
# x2 = list(trainldr_other)
print(sum([torch.sum(k[0]).item() for k in x1]))
import sys

sys.exit()
import pdb

pdb.set_trace()

# print([torch.equal(k1,k2) for k1, k2 in i1,i2 for i1,i2 in zip(x1, x2)])

for i1, i2 in zip(x1, x2):
    for k1, k2 in zip(i1, i2):
        print(torch.equal(k1, k2))
        import pdb

        pdb.set_trace()
        break
    break

# x1 = next(iter(trainldr_other))
# x1 = next(iter(trainldr))
# x2=iter(trainldr_other).next()
# x2 = next(iter(trainldr))

# print(x1[0][0][0])
# print(x1[-1][-1], 'id', id(trainldr))
# print(x1[-1][-1], 'id', id(x1))


# print(x1[0][0][0], '\n', x2[0][0][0])
# print(x2[0][0][0], '\n', x1[0][0][0])

# import pdb; pdb.set_trace()
