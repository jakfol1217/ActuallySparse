import pytest
import torch
from torch.nn.modules import Linear
from actuallysparse import layers
from actuallysparse.layers import SparseLayer


@pytest.mark.parametrize(
    "constructor",
    [layers.new_random_basic_coo, layers.new_random_basic_csr]
)
def test_initialization(constructor):
    assert torch.is_same_size(
        constructor(3, 3).weights.data,
        torch.zeros(3, 3)
    )


@pytest.mark.parametrize(
    "size",
    [[1, 3], [3, 1], [2, 3], [3, 2]]
)
def test_forward_size(size):
    layer = SparseLayer(size[0], size[1])
    data = torch.rand(size[0])
    assert layer.forward(data).size()[0] == size[1]


def test_compare_linear():
    linear = Linear(3, 4)
    sparse = layers.new_random_basic_coo(3, 4)

    sparse.assign_new_weights(
        linear.weight.data,
        linear.bias.data
    )

    data = torch.tensor([1., 2., 3.])
    assert (linear.forward(data) == sparse.forward(data)).all()
