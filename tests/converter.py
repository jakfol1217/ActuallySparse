import pytest
from torch.nn.modules import Linear
import torch
from actuallysparse import converter
from actuallysparse.layers import SparseLayer


@pytest.mark.parametrize(
    "layer_constructor, conversion_target",
    [
        (constructor, target)
        for constructor in [Linear, SparseLayer, lambda x, y: SparseLayer(x, y, csr_mode=True)]
        for target in ["dense", "coo", "csr"]
    ]
)
def test_convert_range(layer_constructor, conversion_target):
    data = torch.tensor([[1., 2., 3.]])

    original = layer_constructor(3, 4)
    converted = converter.convert(original, conversion_target)

    assert (original.forward(data) == converted.forward(data)).all


def test_unknown_target():
    with pytest.raises(TypeError):
        converter.convert(Linear(1, 1), "not_a_type")
