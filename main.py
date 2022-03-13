import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse

from models.models import AVModel
from data.data import load_cifar_10

import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--no-wandb', action='store_true')
args = parser.parse_args()

if not args.no_wandb:
    wandb.init(project="torch-cnn", entity="joeljosephjin", config=args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load cifar-10
trainloader, testloader, classes = load_cifar_10(args.batch_size)
net = AVModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum) # 0.001, 0.9


for epoch in range(args.epochs):  # loop over the dataset multiple times; 4

    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # calc accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        acc = correct / total

        # print statistics
        if not args.no_wandb:
            wandb.log({'loss':loss.item(), 'accuracy':acc})
        running_loss += loss.item()
        running_acc += acc
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f} acc: {running_acc / 2000:.3f}')
            running_loss = 0.0
            running_acc = 0.0

print('Finished Training')


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct // total
print(f'Accuracy of the network on the 10000 test images: {test_accuracy} %')
if not args.no_wandb:
    wandb.run.summary["test_accuracy"] = test_accuracy


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')