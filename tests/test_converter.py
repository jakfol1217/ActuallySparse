import pytest
from torch import nn
from torch.nn.modules import Linear
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


def test_model_converter(request):

    model = nn.Sequential(
        Linear(30, 5),
        Linear(5, 4),
        new_random_basic_coo(4, 1)
    )
    converter.convert_model(model, Linear, 'coo')
    assert all(SparseLayer == type(child) for child in model.children())

