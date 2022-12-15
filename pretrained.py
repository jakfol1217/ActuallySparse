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

class TransformedVgg(nn.Module):
    def __init__(self, model, prev_out_features, new_out_features):
        super(TransformedVgg, self).__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        self.extra_layer = nn.Linear(prev_out_features, new_out_features)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.extra_layer(x)
        return x

def get_pretrained_transformed_vgg(database_name):
    model = torchvision.models.vgg11_bn(weights=torchvision.models.VGG11_BN_Weights.IMAGENET1K_V1)
    if database_name == "cifar10":
        model = TransformedVgg(model, 1000, 10)
    elif database_name == "cifar100":
        model = TransformedVgg(model, 1000, 100)
    elif database_name == "caltech256":
        model = TransformedVgg(model, 1000, 257) #lub 256
    else:
        raise Exception("Unknown database name passed")
    return model

# def fit_one_cycle(epochs, max_lr, model, train_loader, weight_decay=0, grad_clip=None, opt_func=torch.optim.Adam):
#
#     # Set up custom optimizer with weight decay
#     optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
#     # Set up one-cycle learning rate scheduler
#     sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
#                                                 steps_per_epoch=len(train_loader))
#
#     for epoch in range(epochs):
#         # Training Phase
#         model.train()
#         print(f"Epoch:{epoch + 1}")
#         for batch in train_loader:
#             vals, labels = batch
#             out = model(vals)
#             loss = torch.nn.functional.cross_entropy(out, labels)
#             loss.backward()
#
#             # Gradient clipping
#             if grad_clip:
#                 nn.utils.clip_grad_value_(model.parameters(), grad_clip)
#
#             optimizer.step()
#             optimizer.zero_grad()
#
#             # Record & update learning rate
#             sched.step()
#
#
# fit_one_cycle(3, 0.001, transformed_model, dataloader_train, weight_decay=0.01,
#               grad_clip = 0.1, opt_func=torch.optim.Adam)