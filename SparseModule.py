from typing import Any

import torch
from torch import Tensor
from torch import sparse
import torch.nn as nn


# Klasa implementująca samą warstwę, tzn. m.in. przechowywanie paramterów
class SparseLayer(nn.Module):
    def __init__(self):
        super(SparseLayer, self).__init__()

    def forward(self):
        pass


# implementacja funkcjonalności warstwy, a więc przejścia "w przód" oraz "w tył"
class SparseModuleFunction(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx, x: Tensor, weights: Tensor, bias: Tensor) -> Any:
        out = sparse.mm(weights, x) + bias
        ctx.save_for_backward(out, weights)
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass
