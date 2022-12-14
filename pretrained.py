# File based on https://github.com/huyvnphan/PyTorch_CIFAR10/ and https://blog.paperspace.com/vgg-from-scratch-pytorch/

import torch
import torch.nn as nn
import torchvision


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_vgg11_bn_layers():
    cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg11_bn(device="cpu", weights_path = ".weights/state_dicts/vgg11_bn.pt"):
    model = VGG(make_vgg11_bn_layers())
    state_dict = torch.load(
        weights_path, map_location=device
    )
    model.load_state_dict(state_dict)
    return model

def load_cifar10_dataloaders():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_train = torchvision.datasets.CIFAR10(".data", download=True, transform=transform)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16)
    dataset_test = torchvision.datasets.CIFAR10(".data", download=True, train=False, transform=transform)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16)
    return dataloader_train, dataloader_test

def load_cifar100_dataloaders():
    transform = torchvision.transforms.Compose([
        #transforms.Resize((227,227)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset_train = torchvision.datasets.CIFAR100(".data", download=True, transform=transform)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64)
    dataset_test = torchvision.datasets.CIFAR100(".data", download=True, train=False, transform=transform)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64)
    return dataloader_train, dataloader_test

def load_caltech256_dataloaders():
    transform = torchvision.transforms.Compose([
        #torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = torchvision.datasets.Caltech256(".data", download=True, transform=transform)
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [30000, 607])
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16)
    return dataloader_train, dataloader_test