import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import torch.utils.data as data_utils


def load_cifar_10_other(augment=True, batch_size=64):
    # print('augment:', augment, 'batch_size:', batch_size) # true, 64
    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                         transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transform_test),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    # print('trainloader.data.shape:', train_loader.dataset)
    
    return train_loader, val_loader

def load_cifar_10(batch_size=64, perc_size=1):
    # mean = [0.49139968, 0.48215841, 0.44653091]
    mean = [x/255.0 for x in [125.3, 123.0, 113.9]]
    # mean = [0.5, 0.5, 0.5]
    # stdv = [0.24703223, 0.24348513, 0.26158784]
    stdv = [x/255.0 for x in [63.0, 62.1, 66.7]]
    # stdv = [0.5, 0.5, 0.5]
    train_transform = transforms.Compose([
        # transforms.RandomRotation(10),
        # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, stdv)])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, stdv)])
    
    batch_size = batch_size # 4 

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    # trainset = data_utils.Subset(trainset, torch.arange(int(trainset.data.shape[0]*perc_size)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    # print('trainloader.data.shape:', trainloader.dataset)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    # testset = data_utils.Subset(testset, torch.arange(int(testset.data.shape[0]*perc_size)))
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=1, pin_memory=True)
    # print('testloader.data.shape:', testloader.dataset)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # return trainloader, testloader, classes
    return trainloader, testloader

def load_mnist(batch_size=128, perc_size=1):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset1 = torchvision.datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = torchvision.datasets.MNIST('../data', train=False,
                       transform=transform)
    
    trainloader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = [str(i) for i in range(10)] # 0, 1, ... ,10
    
    return trainloader, testloader, classes

