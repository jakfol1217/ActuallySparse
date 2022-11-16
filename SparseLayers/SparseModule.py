from typing import Any

import torch
import numpy as np
from torch import Tensor
from torch import sparse
from torch import sparse_coo_tensor
import torch.nn as nn


# Klasa implementująca samą warstwę, tzn. m.in. przechowywanie paramterów
class SparseLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, csr_mode=False):
        super(SparseLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #Inicjalizacja wag
        weights = torch.FloatTensor(in_features, out_features).uniform_(-1, 1)
        weights[torch.where(abs(weights) <= 0.5)] = 0
        if csr_mode:
            weights = weights.to_sparse_csr()
        else:
            weights = weights.to_sparse_coo()
        self.weights = nn.Parameter(weights)
        #Inicjalizacja biasu
        self.bias = None
        if bias:
            bias = torch.rand(out_features)
            self.bias = nn.Parameter(bias)

    def forward(self, in_values: Tensor):
        if not torch.is_tensor(in_values):
            raise TypeError("Input must be a Tensor")
        if self.in_features not in in_values.size():
            raise Exception("Input values shape does not match")
        if in_values.size()[0] != self.in_features:
            in_values = in_values.t()
        print(in_values.size())
        out = sparse.mm(self.weights.t(), in_values)
        if self.bias is not None:
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
        # out = sparse.mm(weights, x) + bias
        # ctx.save_for_backward(out, weights)

    # return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass
