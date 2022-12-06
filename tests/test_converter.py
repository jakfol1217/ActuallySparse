import pytest
from torch import nn
from torch.nn.modules import Linear, Sequential
import torch
from actuallysparse import converter
from actuallysparse.layers import new_random_basic_coo, new_random_basic_csr, SparseLayer



@pytest.mark.parametrize(
    "layer_constructor, conversion_target",
    [
        (constructor, target)
        for constructor in [Linear, new_random_basic_coo, new_random_basic_csr]
        for target in ["dense", "coo", "csr"]
    ]
)
def test_convert_range(layer_constructor, conversion_target):
    data = torch.tensor([[1., 2., 3.]])

    original = layer_constructor(3, 4)
    converted = converter.convert(original, conversion_target)

    assert torch.allclose(original.forward(data), converted.forward(data))

@pytest.mark.parametrize(
    "layer_constructor, conversion_target, size",
    [
        (constructor, target, size)
        for constructor in [Linear, new_random_basic_coo, new_random_basic_csr]
        for target in ["dense", "coo", "csr"]
        for size in [(3, 3), (3, 4)]
    ]
)
def test_convert_weights(layer_constructor, conversion_target, size):
    original = layer_constructor(size[0], size[1])
    converted = converter.convert(original, conversion_target)

    assert torch.allclose(original.weight.to_dense(), converted.weight.to_dense())


def test_unknown_target():
    with pytest.raises(TypeError):
        converter.convert(Linear(1, 1), "not_a_type")


@pytest.mark.parametrize("conversion_target", ["dense", "coo", "csr"])
def test_convert_with_mask(conversion_target):
    mask = torch.tensor([[1., 0., 0.], [0., 0., 0.], [1., 1., 1.], [0., 0., 1.]])
    original = Linear(3, 4)

    new = converter.convert(original, conversion_target, mask)

    assert pytest.approx(0.) == new.weight.data[1, 1]
    assert original.weight.data[1, 1] != 0.


def test_model_converter():
    model = nn.Sequential(
        Linear(30, 5),
        Linear(5, 4),
        new_random_basic_coo(4, 1)
    )
    model = converter.convert_model(model, Linear, 'coo')
    assert all(SparseLayer == type(child) for child in model.children())


def test_model_converter_recursion():
    model = Sequential(
        Linear(30, 5),
        Sequential(
            Linear(5, 4),
            Linear(4, 6),
            Linear(6, 4)
        ),
        Linear(5, 4),
        new_random_basic_coo(4, 1)
    )
    model = converter.convert_model(model, Linear, 'coo')
    assert all(SparseLayer == type(child) for child in model[1].children())


def test_converted_model_forward():
    loss_fn = nn.CrossEntropyLoss()

    model = nn.Sequential(
        Linear(3, 5),
        Linear(5, 4),
        new_random_basic_coo(4, 1)
    )

    data = torch.Tensor([[1., 2., 3.]])

    output_dense = model(data)
    model = converter.convert_model(model, nn.Linear, 'coo')
    output_sparse = model(data)
    assert torch.allclose(output_dense, output_sparse)

