import wandb
import torch
from torch import nn, optim


class ClassifierPipeline():

    def __init__(self, args=None, net=None, datatuple=None):

        self.args = args
        
        if not self.args.no_wandb:
            wandb.init(project="torch-cnn", entity="joeljosephjin", config=args)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

        torch.autograd.set_detect_anomaly(True)
        
        # load cifar-10
        self.trainloader, self.testloader, self.classes = datatuple
        self.net = net().to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum) # 0.001, 0.9

    def train(self, epochs=3):
        for epoch in range(self.args.epochs):  # loop over the dataset multiple times; 4

            running_loss = 0.0
            running_acc = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
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
                if i % 500 == 499:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f} acc: {running_acc / 2000:.3f}')
                    running_loss = 0.0
                    running_acc = 0.0

        print('Finished Training')

    def test(self):

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
