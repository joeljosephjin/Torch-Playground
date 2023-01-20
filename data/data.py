import torch
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as data_utils


def load_cifar_10(batch_size=4, perc_size=1):
    mean = [0.49139968, 0.48215841, 0.44653091]
    # mean = [0.5, 0.5, 0.5]
    stdv = [0.24703223, 0.24348513, 0.26158784]
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
    trainset = data_utils.Subset(trainset, torch.arange(int(trainset.data.shape[0]*perc_size)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    print('trainloader.data.shape:', trainloader.dataset.dataset)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testset = data_utils.Subset(testset, torch.arange(int(testset.data.shape[0]*perc_size)))
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    print('testloader.data.shape:', testloader.dataset.dataset)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

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

def subset_dataset(dataset, test_classes, train_classes):
    """
    exclude the test_classes to create trainloader
    """
    test_idx = sum(dataset.targets==i for i in test_classes).bool().nonzero().flatten()
    train_idx = sum(dataset.targets==i for i in train_classes).bool().nonzero().flatten()
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    test_subset = torch.utils.data.Subset(dataset, test_idx)
    return train_subset, test_subset

def load_fewshot_mnist(batch_size=128, perc_size=1, test_classes=[0, 1, 2]):
    """
    Join both train and test sets
    Segregate into two sets on the basis of select classes
    """
    
    classes = [i for i in range(10)] # 0, 1, ... , 10
    
    train_classes = [clas for clas in classes if clas not in test_classes]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset1 = torchvision.datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = torchvision.datasets.MNIST('../data', train=False,
                       transform=transform)
    
    dataset1_train, dataset1_test = subset_dataset(dataset1, test_classes, train_classes)
    dataset2_train, dataset2_test = subset_dataset(dataset2, test_classes, train_classes)
    
    trainset = torch.utils.data.ConcatDataset([dataset1_train, dataset2_train])
    testset = torch.utils.data.ConcatDataset([dataset1_test, dataset2_test])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

#     for i, (x, y) in enumerate(trainloader):
#         print(y)
#         if i>=30:
#             break
    
#     print(y.unique())
#     import pdb; pdb.set_trace()
    
    return trainloader, testloader, test_classes
