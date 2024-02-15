"""
import joblib; joblib.dump(self.net, 'model.pkl')
"""
import wandb
import torch
from torch import nn, optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import importlib
from time import time

import torch.nn as nn
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
import torch.optim as optim

import argparse

from models.models import *
from models.shufflenet import ShuffleNet
from data.data import *
from utils import set_seed, accuracy_densenet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--model', type=str, default='SimpleModel', help="SimpleModel or AVModel,..")
    parser.add_argument('--dataset', type=str, default='cifar_10', help="cifar_10 or mnist,..")
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--perc-size', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--save-interval', type=int, default=6)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--resume-from-saved', type=str, default=None, help="name of the exp to load from")
    parser.add_argument('--save-as', type=str, default='', help="a name for the model save file")
    args = parser.parse_args()
    
    return args


class ClassifierPipeline():

    def __init__(self, args=None, net=None, datatuple=None):

        self.args = args
        
        if self.args.use_wandb:
            wandb.init(project="torch-cnn", entity="joeljosephjin", config=self.args)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.autograd.set_detect_anomaly(True)
        
        # load cifar-10
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # model = DenseNet3(40, 10, 12, 1.0, False, 0)
        self.trainloader, self.testloader = load_cifar_10_other()
        # self.trainloader, self.testloader = load_cifar_10()
        
        if self.args.resume_from_saved:
            self.net = self.load_model(filename=self.args.model+self.args.resume_from_saved, modelname=self.args.model)
        else:
            self.net = net().to(self.device)
            
        self.net.train()
        
        model_parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
        print(f'Number of Parameters: {params/int(1e6):.2f}M')
            
        self.criterion = nn.CrossEntropyLoss()
        print('args.lr:', self.args.learning_rate, self.args.momentum, self.args.weight_decay)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum, nesterov=True, weight_decay=self.args.weight_decay) # 0.001, 0.9

    def train(self, epochs=3):
        self.start_time = time()
        logs_interval = 100
        
        for epoch in range(self.args.epochs):  # loop over the dataset multiple times; 4
            self.net.train()
            self.adjust_learning_rate(epoch)

            running_loss = []
            running_acc = []
            running_accs_densenet = []
            for i, data in enumerate(self.trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data[0].cuda(non_blocking=True)
                labels =  data[1].to(self.device)


                # forward + backward + optimize
                params_list=list(self.net.parameters())
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # calc accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                total = labels.size(0)
                acc = correct / total
                
                # calc acc_densenet
                acci_densenet = accuracy_densenet(outputs.data, labels, topk=(1,))[0]

                running_loss.append(loss.item())
                running_acc.append(acc)
                running_accs_densenet.append(acci_densenet)
                if i % logs_interval == 0:# and i != 0:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {sum(running_loss) / len(running_loss):.3f} acc: {sum(running_acc) / len(running_loss):.3f} acc_densenet: {sum(running_accs_densenet) / len(running_accs_densenet):.3f} time: {(time() - self.start_time) / 60:.2f} minutes')
                    running_loss = []
                    running_acc = []
                
            if self.args.use_wandb:
                wandb.log({'train_loss':loss.item(), 'train_acc':acci_densenet.item(), 'model_param':list(self.net.parameters())[0][0][0][0][0].item()})

            if epoch % self.args.log_interval == 0:
                self.test(epoch=epoch)
            # if epoch % self.args.save_interval == 0:
            #     self.save_model(model=self.net, filename=self.args.model+self.args.save_as)
            
        print('Finished Training')

    def test(self, epoch):
        correct = 0
        total = 0
        accs_densenet = []
        self.net.eval()
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
                
                # calc acc_densenet
                acci_densenet = accuracy_densenet(outputs.data, labels, topk=(1,))[0]
                accs_densenet.append(acci_densenet)

        test_accuracy = 100 * correct // total
        acc_densenet = sum(accs_densenet)/len(accs_densenet)
        print(f'Accuracy of the network on the 10000 test images: {test_accuracy} %, acc_densenet is {acc_densenet} %')
        if self.args.use_wandb:
            # wandb.log({'valid_acc': test_accuracy, 'valid_acc_densenet': acc_densenet})
            wandb.log({'valid_acc': acc_densenet})


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
        path = f'saved/{filename}.pth'
        torch.save(model.state_dict(), path)
        print(f'Model saved as {path} ...')
        
    def load_model(self, filename, modelname):
        path = f'saved/{filename}.pth'
        models_mod = importlib.import_module(f'models.models')
        model_class = getattr(models_mod, modelname)
        model = model_class()
        state_dict = torch.load(path)
        
        # state_dict = state_dict['net']
        # model = torch.nn.DataParallel(model)
        
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        print(f'Model loaded successfully from {path} ...')
        return model
    
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
        lr = args.learning_rate * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
        if args.use_wandb:
            wandb.log({'learning_rate':lr})
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

            
if __name__=="__main__":
    args = get_args()
    set_seed(args.seed)
    cudnn.benchmark = True
    print('Loading Dataset..')
    load_dataset_fn = eval(f'load_{args.dataset}')
    datatuple = load_dataset_fn(batch_size=args.batch_size, perc_size=args.perc_size)
    print('Loading Model..')

    pipeline1 = ClassifierPipeline(args=args, net=eval(args.model), datatuple=datatuple)

    print('Starting Training..')
    s = time()
    pipeline1.train(epochs=args.epochs)
    print(f'{time()-s} taken for training...')
    # pipeline1.test()