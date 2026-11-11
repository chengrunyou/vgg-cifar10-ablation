"""CIFAR-10 数据加载与预处理。"""

import os

import torch
import torchvision
import torchvision.transforms as transforms

__all__ = ["get_dataloaders", "CIFAR10_CLASSES"]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(data_dir="./data", batch_size=128, num_workers=0):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    png_train = os.path.join(data_dir, "cifar10_full", "cifar10", "train")
    png_test = os.path.join(data_dir, "cifar10_full", "cifar10", "test")

    if os.path.isdir(png_train) and os.path.isdir(png_test):
        train_set = torchvision.datasets.ImageFolder(root=png_train, transform=train_transform)
        test_set = torchvision.datasets.ImageFolder(root=png_test, transform=test_transform)
    else:
        train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
