from typing import Any

import torch
import numpy as np
from torch import Tensor
from torch import sparse
from torch import sparse_coo_tensor
import torch.nn as nn


# Klasa implementująca samą warstwę, tzn. m.in. przechowywanie paramterów
class SparseLayer(nn.Module):
    def __init__(self, size_in, size_out, bias=True):
        super(SparseLayer, self).__init__()
        self.size_in = size_in
        self.size_out = size_out
        weights = torch.rand(size_in, size_out).to_sparse_coo()
        #weights = sparse_coo_tensor(size=(size_in, size_out))
        self.weights = nn.Parameter(weights)
        self.bias = None
        if bias:
            bias = torch.rand(size_out)
            self.bias = nn.Parameter(bias)

    def forward(self, input):
        out = sparse.mm(self.weights.t(), input)
        if self.bias:
            torch.add(out, self.bias)
        return out


# implementacja funkcjonalności warstwy, a więc przejścia "w przód" oraz "w tył"
class SparseModuleFunction(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        pass
        #out = sparse.mm(weights, x) + bias
        #ctx.save_for_backward(out, weights)
       #return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass
