import pytest
import torch
from torch import nn
from torch.autograd import Variable

from actuallysparse.layers import new_random_basic_coo, new_random_basic_csr

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

