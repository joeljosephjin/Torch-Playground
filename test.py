from data.data import load_cifar_10_other, load_cifar_10
import torch
from utils import set_seed
import models.densenet3 as dn


set_seed(42)

train_loader, val_loader = load_cifar_10_other()
# train_loader, val_loader, _ = load_cifar_10()
# x = list(train_loader)
# for i in range(5):
#     xi = x[i][0][0][0][0][0]
#     print('xi:', xi)
#     import pdb; pdb.set_trace()
    
model = dn.DenseNet3(40, 10, 12, 1.0, False, 0)
    
for i, (input, target) in enumerate(train_loader):
    input_var = torch.autograd.Variable(input)
    print('input_var:', input_var[0][0][0][0])
    import pdb; pdb.set_trace()
    