import torch
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as data_utils


def load_cifar_10(batch_size=4, perc_size=1):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = batch_size # 4 

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset = data_utils.Subset(trainset, torch.arange(int(trainset.data.shape[0]*perc_size)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    print('trainloader.data.shape:', trainloader.dataset.dataset)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testset = data_utils.Subset(testset, torch.arange(int(testset.data.shape[0]*perc_size)))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    print('testloader.data.shape:', testloader.dataset.dataset)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

