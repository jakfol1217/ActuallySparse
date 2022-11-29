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
        weight = torch.FloatTensor(out_features, in_features).uniform_(-1, 1)
        weight[torch.where(abs(weight) <= 0.5)] = 0
        if csr_mode:
            weight = weight.to_sparse_csr()
            values = weight.values()
            self.register_buffer('row_indices', weight.crow_indices())
            self.register_buffer('col_indices', weight.col_indices())
        else:
            weight = weight.to_sparse_coo()
            values = weight.values()
            self.register_buffer('indices', weight.indices())
        self.values = nn.Parameter(values)
        # Inicjalizacja biasu
        if bias:
            bias = torch.rand(out_features)
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, in_values):
        if not torch.is_tensor(in_values):
            raise TypeError("Input must be a Tensor")
        if len(in_values.size()) == 1:
            in_values = in_values.view(-1, 1)
        if self.in_features not in in_values.size():
            raise Exception("Input values shape does not match")
        if in_values.size()[1] != self.in_features:
            in_values = in_values.t()
        if not self.csr_mode:
            weight = torch.sparse_coo_tensor(values=self.values, indices=self.indices,
                                             size=(self.out_features, self.in_features)).to_dense()
        else:
            weight = torch.sparse_csr_tensor(crow_indices=self.row_indices, col_indices=self.col_indices,
                                             values=self.values, size=(self.out_features, self.in_features)).to_dense()
        out = torch.mm(in_values, weight.t())
        if self.bias is not None:
            out = torch.add(out, self.bias)
        return out

    # Funkcja służąca do nadawania nowych wag, głównie przy inicjalizacji
    # ma automatycznie przekształcać na reprezentację rzadką
    def assign_new_weight(self, new_weight, bias=None):
        if not torch.is_tensor(new_weight):
            raise TypeError("New weight must be a Tensor")
        if new_weight.size() != torch.Size([self.out_features, self.in_features]):
            if new_weight.t().size() == torch.Size([self.out_features, self.in_features]):
                new_weight = new_weight.t()
            else:
                raise Exception("Weight shape mismatch")
        if bias is not None and bias.size() != torch.Size([self.out_features]):
            raise Exception("Bias shape mismatch")

        if bias is not None:
            self.bias = nn.Parameter(bias)

        if self.csr_mode:
            weight = new_weight.to_sparse_csr()
            self.values = nn.Parameter(weight.values())
            self.register_buffer('row_indices', weight.crow_indices())
            self.register_buffer('col_indices', weight.col_indices())
            return
        weight = new_weight.to_sparse_coo()
        self.values = nn.Parameter(weight.values())
        self.register_buffer('indices', weight.indices())

    # Ustawia k procent najmniejszych wartości na 0
    def prune_smallest_values(self, k=0.1, remove_zeros=True):
        if k < 0 or k > 1:
            raise Exception("K must be a value between 0 and 1")
        with torch.no_grad():
            values = self.values
            values[abs(values) < torch.quantile(abs(values), q=k)] = 0
        if remove_zeros:
            self.remove_zeros_from_weight()

    # Usuwa zera z listy wartości wagi, nie działa dla reprezentacji CSR
    # Tworzy nowy tensor rzadki i zapisuje na miejsce starego
    def remove_zeros_from_weight(self):
        if self.csr_mode:
            raise Exception("Cannot remove zeroes with csr mode on")
        mask = self.values.nonzero().view(-1)
        self.values = nn.Parameter(self.values.index_select(0, mask))
        self.indices = self.indices.index_select(1, mask)

    @property
    def weight(self):
        if self.csr_mode:
            weight = torch.sparse_csr_tensor(crow_indices=self.row_indices, col_indices=self.col_indices,
                                             values=self.values, size=(self.out_features, self.in_features))
        else:
            weight = torch.sparse_coo_tensor(indices=self.indices, values=self.values,
                                             size=(self.out_features, self.in_features))
        return weight.t()

# implementacja funkcjonalności warstwy, a więc przejścia "w przód" oraz "w tył"
"""
class SparseModuleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, in_values, weight, bias=None):
        ctx.save_for_backward(in_values, weight, bias)
        out = sparse.mm(weight.t(), in_values).t()
        if bias is not None:
            out = torch.add(out, bias)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        in_values, weight, bias = ctx.saved_tensors
        grad_in_values = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_in_values = sparse.mm(weight, grad_output.t())
        if ctx.needs_input_grad[1] and weight.is_sparse_csr:
            grad_weight = torch.mm(in_values, grad_output).to_sparse_csr()
        elif ctx.needs_input_grad[1] and not weight.is_sparse_csr:
            grad_weight = torch.mm(in_values, grad_output).to_sparse_coo()
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_in_values, grad_weight, grad_bias
"""


def new_random_basic_coo(in_features, out_features, bias=True):
    return SparseLayer(in_features, out_features, bias=bias)


def new_random_basic_csr(in_features, out_features, bias=True):
    return SparseLayer(in_features, out_features, bias=bias, csr_mode=True)
