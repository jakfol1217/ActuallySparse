from typing import Any

import torch
from torch import Tensor
from torch import sparse
import torch.nn as nn


# Klasa implementująca samą warstwę, tzn. m.in. przechowywanie parametrów
class SparseLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, csr_mode=False):
        super(SparseLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.csr_mode = csr_mode
        # Inicjalizacja wag
        weights = torch.FloatTensor(in_features, out_features).uniform_(-1, 1)
        weights[torch.where(abs(weights) <= 0.5)] = 0
        if csr_mode:
            weights = weights.to_sparse_csr()
        else:
            weights = weights.to_sparse_coo()
        self.weights = nn.Parameter(weights)
        # Inicjalizacja biasu
        self.bias = None
        if bias:
            bias = torch.rand(out_features)
            self.bias = nn.Parameter(bias)

    def forward(self, in_values: Tensor):
        if not torch.is_tensor(in_values):
            raise TypeError("Input must be a Tensor")
        if len(in_values.size()) == 1:
            in_values = in_values.view(-1, 1)
        if self.in_features not in in_values.size():
            raise Exception("Input values shape does not match")
        if in_values.size()[0] != self.in_features:
            in_values = in_values.t()
        out = sparse.mm(self.weights.t(), in_values)
        if self.bias is not None:
            out = torch.add(out.t(), self.bias)
        return out

    # Funkcja służąca do nadawania nowych wag, głównie przy inicjalizacji
    # ma automatycznie przekształcać na reprezentację rzadką
    def assign_new_weights(self, new_weights, bias=None):
        if not torch.is_tensor(new_weights):
            raise TypeError("New weights must be a Tensor")
        if new_weights.size() != torch.Size([self.in_features, self.out_features]):
            if new_weights.t().size() == torch.Size([self.in_features, self.out_features]):
                new_weights = new_weights.t()
            else:
                raise Exception("Weight shape mismatch")
        if bias is not None and bias.size() != torch.Size([self.out_features]):
            raise Exception("Bias shape mismatch")

        if bias is not None:
            self.bias = nn.Parameter(bias)

        if self.csr_mode:
            self.weights = nn.Parameter(new_weights.to_sparse_csr())
            return
        self.weights = nn.Parameter(new_weights.to_sparse_coo())


# implementacja funkcjonalności warstwy, a więc przejścia "w przód" oraz "w tył"
# TODO
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


def new_random_basic_coo(in_features, out_features, bias=True):
    return SparseLayer(in_features, out_features, bias=bias)


def new_random_basic_csr(in_features, out_features, bias=True):
    return SparseLayer(in_features, out_features, bias=bias, csr_mode=True)
