# File based on https://github.com/huyvnphan/PyTorch_CIFAR10/ and https://blog.paperspace.com/vgg-from-scratch-pytorch/

import torch
import torch.nn as nn
import torchvision

 # Klasa reprezentująca wykorzystywaną architekturę VGG
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

 # Funkcja tworząca część konwolucyjną archiektury VGG11_bn, wykorzystywana w klasie VGG
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

 # Funkcja zwracająca model VGG11_bn
def vgg11_bn(device="cpu", weights_path = ".weights/state_dicts/vgg11_bn.pt"):
    model = VGG(make_vgg11_bn_layers())
    state_dict = torch.load(
        weights_path, map_location=device
    )
    model.load_state_dict(state_dict)
    return model

 # Funkcja zwracająca dane ze zbioru Cifar-10
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

 # Funkcja zwracająca dane ze zbioru Cifar-100
def load_cifar100_dataloaders():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset_train = torchvision.datasets.CIFAR100(".data", download=True, transform=transform)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16)
    dataset_test = torchvision.datasets.CIFAR100(".data", download=True, train=False, transform=transform)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16)
    return dataloader_train, dataloader_test

 # Funkcja zwracająca dane ze zbioru Caltech-256
def load_caltech256_dataloaders():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = torchvision.datasets.Caltech256(".data", download=True, transform=transform)
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [30000, 607])
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16)
    return dataloader_train, dataloader_test

 # Funkcja zwracająca dane ze zbioru Cifar-10 ze zbiorem walidacyjnym
def load_cifar10_dataloaders_validation():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(".data", download=True, transform=transform)
    size_train = 0.9*len(dataset)
    size_val = len(dataset) - size_train
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [int(size_train), int(size_val)])
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16)
    dataset_test = torchvision.datasets.CIFAR10(".data", download=True, train=False, transform=transform)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16)
    return dataloader_train, dataloader_test, dataloader_val

 # Funkcja zwracająca dane ze zbioru Cifar-100 ze zbiorem walidacyjnym
def load_cifar100_dataloaders_validation():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = torchvision.datasets.CIFAR100(".data", download=True, transform=transform)
    size_train = 0.9*len(dataset)
    size_val = len(dataset) - size_train
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [int(size_train), int(size_val)])
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16)
    dataset_test = torchvision.datasets.CIFAR100(".data", download=True, train=False, transform=transform)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16)
    return dataloader_train, dataloader_test, dataloader_val

 # Funkcja zwracająca dane ze zbioru Caltech-256 ze zbiorem walidacyjnym
def load_caltech256_dataloaders_validation():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = torchvision.datasets.Caltech256(".data", download=True, transform=transform)
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [30000, 607])
    size_train = 0.9*len(dataset_train)
    size_val = len(dataset_train) - size_train
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_train, [int(size_train), int(size_val)])
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16)
    return dataloader_train, dataloader_test, dataloader_val


 # Klasa reprezentująca model VGG dostosowany do innych zbiorów danych (innych niż podstawowy w projekcie Cifar-10)
class TransformedVgg(nn.Module):
    def __init__(self, model, new_out_features):
        super(TransformedVgg, self).__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        layers = list(model.classifier.children())
        no_last_layer = layers[:-1]
        prev_in_features = layers[-1].in_features
        self.classifier = nn.Sequential(*no_last_layer, nn.Linear(prev_in_features, new_out_features))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

 # Funkcja zwracająca model VGG dostosowany do wskazanego zbioru danych
def get_pretrained_transformed_vgg(database_name: str):
    model = torchvision.models.vgg11_bn(weights=torchvision.models.VGG11_BN_Weights.IMAGENET1K_V1)
    if database_name == "cifar10":
        model = TransformedVgg(model, 10)
    elif database_name == "cifar100":
        model = TransformedVgg(model, 100)
    elif database_name == "caltech256":
        model = TransformedVgg(model, 257)
    else:
        raise Exception("Unknown database name passed")
    return model

def get_pretrained_transformed_vgg16(database_name: str):
    model = torchvision.models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1)
    if database_name == "cifar10":
        model = TransformedVgg(model, 10)
    elif database_name == "cifar100":
        model = TransformedVgg(model, 100)
    elif database_name == "caltech256":
        model = TransformedVgg(model, 257)
    else:
        raise Exception("Unknown database name passed")
    return model
