# python main.py --no-wandb --epochs 5 --learning-rate 0.01 --perc-size 0.05
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse

from models.models import AVModel, SimpleModel
from models.shufflenet import ShuffleNet
from data.data import load_cifar_10

from pipeline import ClassifierPipeline

import time


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--perc-size', type=float, default=1)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--no-wandb', action='store_true')
args = parser.parse_args()


if __name__=="__main__":
    print('Loading Dataset..')
    datatuple = load_cifar_10(batch_size=args.batch_size, perc_size=args.perc_size)
    print('Loading Model..')

    pipeline1 = ClassifierPipeline(args, AVModel, datatuple)
    # pipeline1 = ClassifierPipeline(args, ShuffleNet, datatuple)
    # pipeline1 = ClassifierPipeline(args, SimpleModel, datatuple)

    print('Starting Training..')
    s = time.time()
    pipeline1.train(epochs=args.epochs)
    print(f'{time.time()-s} taken for training,\n Starting Testing..')
    pipeline1.test()
