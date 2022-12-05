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
        constructor(3, 3).weight.data,
        torch.zeros(3, 3)
    )


@pytest.mark.parametrize(
    "size",
    [[1, 3], [3, 1], [2, 3], [3, 2]]
)
def test_forward_size(size):
    layer = SparseLayer(size[0], size[1])
    data = torch.rand(size[0])
    assert layer.forward(data).size()[1] == size[1]


def test_compare_linear():
    linear = Linear(3, 4)
    sparse = layers.new_random_basic_coo(3, 4)

    sparse.assign_new_weight(
        linear.weight.data,
        linear.bias.data
    )

    data = torch.tensor([[1., 2., 3.]])
    assert (linear.forward(data).size() == sparse.forward(data).size())
    assert torch.allclose(linear.forward(data), sparse.forward(data))


def test_csr_backward():
    sparse = layers.new_random_basic_csr(3, 1)
    data = torch.Tensor([1., 2., 3.])
    out = sparse(data)
    out.backward()
    assert sparse.values.grad is None


@pytest.mark.parametrize(
    "size",
    [[3, 1], [3, 2], [3, 4]]
)
def test_compare_linear_backward(size):
    linear = Linear(size[0], size[1])
    sparse = layers.new_random_basic_coo(size[0], size[1])

    sparse.assign_new_weight(
        linear.weight.data,
        linear.bias.data
    )

    data = torch.tensor([[1., 2., 3.]])

    res_linear = linear.forward(data)
    res_sparse = sparse.forward(data)

    res_linear.sum().backward()
    res_sparse.sum().backward()

    assert torch.allclose(linear.weight.grad, sparse.values.grad.view(size[1], size[0]))

@pytest.mark.parametrize(
    "k, tensor_after_pruning",
        [(0.1, torch.Tensor([[0., 0.2, 0.3, 0.4, 0.5],
                             [0.6, 0.7, 0.8, 0.9, 1.]])),
        (0.33, torch.Tensor([[0., 0., 0., 0.4, 0.5],
                            [0.6, 0.7, 0.8, 0.9, 1.]])),
        (0.5, torch.Tensor([[0., 0., 0., 0., 0.],
                           [0.6, 0.7, 0.8, 0.9, 1.]]))
         ]
)
def test_pruning(k, tensor_after_pruning):
    sparse = layers.new_random_basic_coo(2, 5)
    sparse.assign_new_weight(
        torch.Tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                      [0.6, 0.7, 0.8, 0.9, 1.]])
    )
    sparse.set_k(k)
    sparse.prune_smallest_values()
    assert torch.allclose(sparse.weight.t().to_dense(), tensor_after_pruning)


def test_pruning_reduce_size():
    sparse = layers.new_random_basic_coo(2, 5)
    sparse.assign_new_weight(
        torch.Tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                      [0.6, 0.7, 0.8, 0.9, 1.]])
    )
    sparse.set_k(0.5)
    sparse.prune_smallest_values()
    assert len(sparse.weight.coalesce().values()) == 5


def test_pruning_grad_retention():
    sparse = layers.new_random_basic_coo(3, 5)
    sparse.values.requires_grad_(False)
    sparse.prune_smallest_values()
    assert not sparse.values.requires_grad