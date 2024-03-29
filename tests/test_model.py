import pytest
import torch
from torch import nn
from torch.autograd import Variable

from actuallysparse.layers import new_random_basic_coo, new_random_basic_csr
from actuallysparse.converter import convert_model
from actuallysparse.layers import prune_sparse_model
from models.pretrained import vgg11_bn

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

LAYER_CONSTRUCTORS = [nn.Linear, new_random_basic_coo]


@pytest.fixture
def iris_data():
    iris = load_iris()
    X = iris["data"]
    X = Variable(torch.from_numpy(StandardScaler().fit_transform(X))).float()
    y = iris["target"]
    y = Variable(torch.from_numpy(y)).long()
    return X, y


@pytest.fixture(params=[
    (l1, l2)
    for l1 in LAYER_CONSTRUCTORS
    for l2 in LAYER_CONSTRUCTORS
])
def model_loss_optimizer(request):
    l1, l2 = request.param
    model = nn.Sequential(
        l1(4, 32),
        nn.Sigmoid(),
        l2(32, 3),
        nn.Softmax(dim=1)
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, loss_fn, optimizer


def test_model_improvement(iris_data, model_loss_optimizer):
    X, y = iris_data
    model, loss_fn, optimizer = model_loss_optimizer

    with torch.no_grad():
        y_pred = model(X)
        loss_start = loss_fn(y_pred, y)

    for i in range(100):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred = model(X)
        loss_end = loss_fn(y_pred, y)

    assert (loss_start - loss_end) >= 0.01


def test_training_after_convert(iris_data):
    X, y = iris_data
    loss_fn = nn.CrossEntropyLoss()

    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.Sigmoid(),
        nn.Linear(32, 3),
        nn.Softmax(dim=1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with torch.no_grad():
        y_pred = model(X)
        loss_start = loss_fn(y_pred, y)

    for i in range(50):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model = convert_model(model, nn.Linear, 'coo')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with torch.no_grad():
        y_pred = model(X)
        loss_mid = loss_fn(y_pred, y)

    for i in range(50):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred = model(X)
        loss_end = loss_fn(y_pred, y)

    assert (loss_start - loss_mid) >= 0.01 and (loss_mid - loss_end) >= 0.01


def test_model_pruning(iris_data):
    X, y = iris_data
    loss_fn = nn.CrossEntropyLoss()

    model = nn.Sequential(
        new_random_basic_coo(4, 64),
        nn.Sigmoid(),
        new_random_basic_coo(64, 3),
        nn.Softmax(dim=1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    values_start = model[0].values
    prune_sparse_model(model, X)
    values_mid = model[0].values

    with torch.no_grad():
        y_pred = model(X)
        loss_start = loss_fn(y_pred, y)

    for i in range(100):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred = model(X)
        loss_end = loss_fn(y_pred, y)
    values_end = model[0].values

    assert values_mid.requires_grad
    assert len(values_start) > len(values_mid)
    assert len(values_mid) == len(values_end)
    assert (loss_start - loss_end) >= 0.01

# Sprawdzenie czy funkcje działają na wytrenowanym już modelu
def test_pretrained_model_convert_and_prune():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vgg11_bn(device=device, weights_path="../.weights/state_dicts/vgg11_bn.pt")
    model.classifier = convert_model(model.classifier, nn.Linear, 'coo')
    dummy_input = torch.ones(512)
    values_start = model.classifier[0].values
    prune_sparse_model(model.classifier, dummy_input)
    values_end = model.classifier[0].values
    assert len(values_start) > len(values_end)






