import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse

from models.models import AVModel
from data.data import load_cifar_10

from pipeline import ClassifierPipeline


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--no-wandb', action='store_true')
args = parser.parse_args()


datatuple = load_cifar_10(batch_size=args.batch_size)
pipeline1 = ClassifierPipeline(args, AVModel, datatuple)

pipeline1.train(epochs=args.epochs)
print('...next...')
pipeline1.test()
