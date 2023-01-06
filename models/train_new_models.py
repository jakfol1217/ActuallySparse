import argparse
import torch
from pretrained import get_pretrained_transformed_vgg
from pretrained import load_cifar10_dataloaders, load_cifar100_dataloaders, load_caltech256_dataloaders


training_device = "cuda"

## ------------MODEL INITIALIZATION------------
model_cifar10 = get_pretrained_transformed_vgg('cifar10')
model_cifar100 = get_pretrained_transformed_vgg('cifar100')
model_caltech256 = get_pretrained_transformed_vgg('caltech256')
## ------------MODEL INITIALIZATION END------------


## ------------DATA LOADERS------------
train_cifar10, test_cifar10 = load_cifar10_dataloaders()
train_cifar100, test_cifar100 = load_cifar100_dataloaders()
train_caltech256, test_caltech256 = load_caltech256_dataloaders()
## ------------DATA LOADERS END------------

## ------------TRAINING PARAMETERS------------
criterion = torch.nn.CrossEntropyLoss()
cifar10_optim = torch.optim.Adam(model_cifar10.parameters(), lr=1e-3)
cifar100_optim = torch.optim.Adam(model_cifar100.parameters(), lr=1e-3)
caltech256_optim = torch.optim.Adam(model_caltech256.parameters(), lr=1e-3)
## ------------TRAINING PARAMETERS END------------

## ------------SCRIPT ARGUMENTS------------
parser = argparse.ArgumentParser(description='Script trains VGG11_bn models on 3 datasets: Cifar-10, Cifar-100 and Caltech-256',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('cif10epochs', type=int, help= 'max epochs for cifar-10 dataset')
parser.add_argument('cif100epochs', type=int, help= 'max epochs for cifar-100 dataset')
parser.add_argument('cal256epochs', type=int, help= 'max epochs for caltech-256 dataset')
parser.add_argument('dst', help='destination path for trained weights')
args = vars(parser.parse_args())
## ------------SCRIPT ARGUMENTS END------------

def training_func(model, optimizers, criterion, dataloader, max_epochs, *_args, **_kwargs):
    model.train()
    model.to(training_device)
    torch.cuda.empty_cache()
    for epoch in range(max_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(training_device), labels.to(training_device)
            optimizers.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizers.step()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    training_func(model_cifar10, cifar10_optim, criterion, train_cifar10, max_epochs=args['cif10epochs'])
    training_func(model_cifar100, cifar100_optim, criterion, train_cifar100, max_epochs=args['cif100epochs'])
    training_func(model_caltech256, caltech256_optim, criterion, train_caltech256, max_epochs=args['cal256epochs'])
    torch.save(model_cifar10.state_dict(), args['dst'] + '/cifar10.pt')
    torch.save(model_cifar100.state_dict(), args['dst'] + '/cifar100.pt')
    torch.save(model_caltech256.state_dict(), args['dst'] + '/caltech256.pt')
