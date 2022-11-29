from typing import Any

import torch
from torch import Tensor
from torch import sparse
import warnings
import torch.nn as nn


# Klasa implementująca samą warstwę, tzn. m.in. przechowywanie parametrów
class SparseLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, csr_mode=False, k = 0.1, training = True):
        super(SparseLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.csr_mode = csr_mode
        self.k = k
        self.training = training
        # Inicjalizacja wag
        weight = torch.FloatTensor(out_features, in_features).uniform_(-1, 1)
        weight[torch.where(abs(weight) <= 0.5)] = 0
        if csr_mode:
            weight = weight.to_sparse_csr()
            values = weight.values()
            self.register_buffer('row_indices', weight.crow_indices())
            self.register_buffer('col_indices', weight.col_indices())
            warnings.warn("WARNING: Training is not supported in csr mode")
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
            if not self.training:
                return PruneOnBackward.apply(in_values, self.values, self.indices,
                                             self.bias, self.in_features, self.out_features, self.k)
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
    """
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
    """


    @property
    def weight(self):
        if self.csr_mode:
            weight = torch.sparse_csr_tensor(crow_indices=self.row_indices, col_indices=self.col_indices,
                                             values=self.values, size=(self.out_features, self.in_features))
        else:
            weight = torch.sparse_coo_tensor(indices=self.indices, values=self.values,
                                             size=(self.out_features, self.in_features))
        return weight

    def __repr__(self):
        return f"SparseLayer(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, " \
               f"csr_mode={self.csr_mode}, k={self.k}, training={self.training})"

    def set_k(self, k):
        self.k = k

    def set_training(self, training):
        self.training = training

# implementacja funkcjonalności warstwy, a więc przejścia "w przód" oraz "w tył"


class PruneOnBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, in_values, values, indices, bias, in_features, out_features, k):
        ctx.save_for_backward(in_values, values, bias)
        ctx.indices = indices
        ctx.k = k
        ctx.in_features = in_features
        ctx.out_features = out_features
        weight = torch.sparse_coo_tensor(values=values, indices=indices,
                                         size=(out_features, in_features)).to_dense()
        out = torch.mm(in_values, weight.t())
        if bias is not None:
            out = torch.add(out, bias)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        in_values, values, bias = ctx.saved_tensors
        k = ctx.k
        indices = ctx.indices
        in_features = ctx.in_features
        out_features = ctx.out_features
        grad_in_values = None
        if ctx.needs_input_grad[0]:
            weight = torch.sparse_coo_tensor(indices=indices, values=values,
                                             size=(out_features, in_features)).to_dense()
            grad_in_values = torch.mm(grad_output, weight)

        vals, indxs= prune_smallest_values(values, indices, k)
        values = vals
        indices = indxs
        return grad_in_values, None, None, None, None, None, None


def prune_smallest_values(values, indices, k=0.1, remove_zeros=True):
    if k < 0 or k > 1:
        raise Exception("K must be a value between 0 and 1")
    with torch.no_grad():
        values = values
        values[abs(values) < torch.quantile(abs(values), q=k)] = 0
    if remove_zeros:
        values, indices = remove_zeros_from_weight(values, indices)
    return values, indices

# Usuwa zera z listy wartości wagi, nie działa dla reprezentacji CSR
# Tworzy nowy tensor rzadki i zapisuje na miejsce starego
def remove_zeros_from_weight(values, indices):
    mask = values.nonzero().view(-1)
    values = nn.Parameter(values.index_select(0, mask))
    indices = indices.index_select(1, mask)
    return values, indices


def new_random_basic_coo(in_features, out_features, bias=True):
    return SparseLayer(in_features, out_features, bias=bias)


def new_random_basic_csr(in_features, out_features, bias=True):
    return SparseLayer(in_features, out_features, bias=bias, csr_mode=True)


def new_random_pruning_coo(in_features, out_features, bias=True, k=0.1):
    return SparseLayer(in_features, out_features, bias=bias, k=k, training=False)
