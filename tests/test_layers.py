import pytest
import torch
from actuallysparse.layers import SparseLayer


@pytest.mark.parametrize(
    "size",
    [[1, 3], [3, 1], [2, 3], [3, 2]]
)
def test_forward(size):
    layer = SparseLayer(size[0], size[1])
    data = torch.rand(size[0])
    layer.forward(data)
