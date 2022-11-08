import wandb
import torch
from torch import nn, optim
# python main.py --no-wandb --epochs 5 --learning-rate 0.01 --perc-size 0.05
import torchvision.transforms as transforms
import importlib

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse

from models.models import *
from models.shufflenet import ShuffleNet
from data.data import *

# from pipeline import ClassifierPipeline

import time


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--model', type=str, default='SimpleModel', help="SimpleModel or AVModel,..")
parser.add_argument('--dataset', type=str, default='cifar_10', help="cifar_10 or mnist,..")
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--perc-size', type=float, default=1)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--save-interval', type=int, default=3)
parser.add_argument('--no-wandb', action='store_true')
parser.add_argument('--resume-from-saved', action='store_false')
args = parser.parse_args()

class ClassifierPipeline():

    def __init__(self, args=None, net=None, datatuple=None):

        self.args = args
        
        if not self.args.no_wandb:
            wandb.init(project="torch-cnn", entity="joeljosephjin", config=args)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # if self.device=
        # self.device = torch.device('cpu')

        torch.autograd.set_detect_anomaly(True)
        
        # load cifar-10
        self.trainloader, self.testloader, self.classes = datatuple
        self.net = net().to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum) # 0.001, 0.9

    def train(self, epochs=3):
        logs_interval = 100
        for epoch in range(self.args.epochs):  # loop over the dataset multiple times; 4

            running_loss = 0.0
            running_acc = 0.0
            for i, data in enumerate(self.trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                # import pdb; pdb.set_trace()
                loss.backward()
                self.optimizer.step()

                # calc accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                total = labels.size(0)
                acc = correct / total

                # print statistics
                if not self.args.no_wandb:
                    wandb.log({'loss':loss.item(), 'accuracy':acc})
                running_loss += loss.item()
                running_acc += acc
                if i % logs_interval == 0 and i != 0:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / logs_interval:.3f} acc: {running_acc / logs_interval:.3f}')
                    running_loss = 0.0
                    running_acc = 0.0
                
            if epoch % 3:
                self.test()
            if epoch % self.args.save_interval:
                self.save_model(self.net, self.args.model)
                self.net = self.load_model(self.args.model, self.args.model)

        print('Finished Training')

    def test(self):
        
        # REMOVE THIS | ONLY FOR TEST
        # self.net.load_state_dict(torch.load('../efficient_densenet_pytorch/save/model.dat'))

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # calculate outputs by running images through the network
                outputs = self.net(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct // total
        print(f'Accuracy of the network on the 10000 test images: {test_accuracy} %')
        if not self.args.no_wandb:
            wandb.run.summary["test_accuracy"] = test_accuracy


        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}

        # again no gradients needed
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(inputs)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
            
    def save_model(self, model, filename):
        path = f'save/{filename}.pth'
        torch.save(model.state_dict(), path)
        print(f'Model saved as {path} ...')
        
    def load_model(self, filename, modelname):
        path = f'save/{filename}.pth'
        models_mod = importlib.import_module(f'models.models')
        model_class = getattr(models_mod, modelname)
        model = model_class()
        model.load_state_dict(torch.load(path))
        model = model.to(self.device)
        print(f'Model loaded successfully from {path} ...')
        return model

            
if __name__=="__main__":
    print('Loading Dataset..')
    load_dataset_fn = eval(f'load_{args.dataset}')
    datatuple = load_dataset_fn(batch_size=args.batch_size, perc_size=args.perc_size)
    print('Loading Model..')

    # pipeline1 = ClassifierPipeline(args, AVModel, datatuple)
    pipeline1 = ClassifierPipeline(args, eval(args.model), datatuple)
    # pipeline1 = ClassifierPipeline(args, ShuffleNet, datatuple)
    # pipeline1 = ClassifierPipeline(args, SimpleModel, datatuple)

    print('Starting Training..')
    s = time.time()
    pipeline1.train(epochs=args.epochs)
    print(f'{time.time()-s} taken for training...')
    # pipeline1.test()