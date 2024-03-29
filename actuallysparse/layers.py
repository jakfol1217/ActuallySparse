import math
import torch
import warnings
import torch.nn as nn
import numpy as np
import torch.autograd.profiler as profiler



# Klasa implementująca samą warstwę, tzn. m.in. przechowywanie parametrów
class SparseLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True, csr_mode: bool=False, train_mode: bool = True, k: float=0.05):
        ### OPIS PARAMETRÓW ###
        # in_features - liczba parametrów wejściowych
        # out_feaatures - liczba parametrów wyjściowych (do następnej warstwy)
        # bias - flaga stwierdzająca, czy warstwa ma posiadać wektor bias'u
        # csr_mode - flaga stwierdzająca, czy warstwa ma przechowywać wagi w reprezentacji CSR (true) czy COO (false)
        # train_mode - flaga stwierdzająca, czy warstwa jest w trybie treningu (wiąże się to m.in. z posobem przejścia "w przód"
        # k - wartość stanowiąca, jaki % parametrów warstwy ma zostać usunięty metodą "prune_smallest_values"
        ###
        ### SZCZEGÓŁY DZIAŁANIA ###
        # Klasa reprezentuje warstwę z macierzą wag rozmiaru in_features x out_features oraz opcjonalnym wektorem bias'u
        # Parametry klasy (pojęcie z biblioteki PyTorch) różnią się w zależności od implementacji
        # Wagi inicjalizowane są w sposób losowy, a następnie elementy których absolutna wartość jest niewiększa niż 0.2 są zerowane
        ###
        super(SparseLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.__csr_mode = csr_mode
        self.train_mode = train_mode
        self.k = k
        # Inicjalizacja wag
        weight = torch.FloatTensor(out_features, in_features).uniform_(-1, 1)
        weight[torch.where(abs(weight) <= 0.2)] = 0
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

    # Funkcja reprezentująca "przejście w przód"
    def forward(self, in_values):
        if not torch.is_tensor(in_values):
            raise TypeError("Input must be a Tensor")
        if len(in_values.size()) == 1:
            in_values = in_values.view(-1, 1)
        if self.in_features not in in_values.size():
            raise Exception("Input values shape does not match")
        if in_values.size()[1] != self.in_features:
            in_values = in_values.t()
        if not self.__csr_mode:
            weight = torch.sparse_coo_tensor(values=self.values, indices=self.indices,
                                             size=(self.out_features, self.in_features))
        else:
            weight = torch.sparse_csr_tensor(crow_indices=self.row_indices, col_indices=self.col_indices,
                                             values=self.values, size=(self.out_features, self.in_features))
        if self.train_mode and self.__csr_mode:
            raise Exception("CSR mode does not support training")
        if self.train_mode and not self.__csr_mode:
            weight = weight.to_dense()
            out = torch.mm(in_values, weight.t())
            if self.bias is not None:
                out = torch.add(out, self.bias)
        elif self.train_mode is False:
            with profiler.record_function("SPARSE PASS"):
                out = torch.sparse.mm(weight, in_values.t()).t()
                if self.bias is not None:
                    out = torch.add(out, self.bias)

        return out

    # Funkcja służąca do nadawania nowych wag, głównie przy inicjalizacji
    # Funkcja ta ma automatycznie przekształcać na reprezentację rzadką
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

        if self.__csr_mode:
            weight = new_weight.to_sparse_csr()
            self.values = nn.Parameter(weight.values())
            self.register_buffer('row_indices', weight.crow_indices())
            self.register_buffer('col_indices', weight.col_indices())
            return
        weight = new_weight.to_sparse_coo()
        self.values = nn.Parameter(weight.values())
        self.register_buffer('indices', weight.indices())

    # Ustawia k procent najmniejszych wartości na 0
    def prune_smallest_values(self):
        num_elements_to_prune = math.floor(self.k * self.in_features * self.out_features)
        with torch.no_grad():
            values = abs(self.values.cpu().numpy())
            num_non_zero_values = len(values)
            if num_non_zero_values <= num_elements_to_prune:
                self.values[self.values!=0] = 0
            else:
                indexes_to_prune = np.argpartition(values, num_elements_to_prune)[:num_elements_to_prune]
                self.values[np.sort(indexes_to_prune)] = 0
        self.remove_zeros_from_weight()

    # Usuwa zera z listy wartości wagi, nie działa dla reprezentacji CSR
    # Tworzy nowy tensor rzadki i zapisuje na miejsce starego
    def remove_zeros_from_weight(self):
        if self.__csr_mode:
            raise Exception("Cannot remove zeroes with csr mode on")
        mask = self.values.nonzero().view(-1)
        require_grad = self.values.requires_grad
        self.values = nn.Parameter(self.values.index_select(0, mask))
        self.values.requires_grad_(require_grad)
        indices = self.indices.index_select(1, mask)
        self.register_buffer('indices', indices)

    def switch_to_CSR(self):
        if self.__csr_mode:
            return
        weight = torch.sparse_coo_tensor(values=self.values, indices=self.indices,
                                             size=(self.out_features, self.in_features))
        weight = weight.to_sparse_csr()
        with torch.no_grad():
            self.register_buffer('row_indices', weight.crow_indices())
            self.register_buffer('col_indices', weight.col_indices())
            self.values = nn.Parameter(weight.values())
        self.train_mode = False
        self.__csr_mode = True


    def switch_to_COO(self):
        if not self.__csr_mode:
            return
        weight = torch.sparse_csr_tensor(crow_indices=self.row_indices, col_indices=self.col_indices,
                                         values=self.values, size=(self.out_features, self.in_features))
        weight = weight.to_sparse_coo()
        with torch.no_grad():
            self.register_buffer('indices', weight.indices())
            self.values = nn.Parameter(weight.values())
        self.__csr_mode = False

    def is_CSR(self):
        return self.__csr_mode


    @property
    def weight(self):
        if self.__csr_mode:
            weight = torch.sparse_csr_tensor(crow_indices=self.row_indices, col_indices=self.col_indices,
                                             values=self.values, size=(self.out_features, self.in_features))
        else:
            weight = torch.sparse_coo_tensor(indices=self.indices, values=self.values,
                                             size=(self.out_features, self.in_features))
        return weight

    def __repr__(self):
        return f"SparseLayer(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, csr_mode={self.__csr_mode}, k={self.k})"

    # Funkcje ustawiające parametry sieci
    def set_k(self, k):
        if k < 0 or k > 1:
            raise Exception("K must be a value between 0 and 1")
        self.k = k
    def train(self, mode = True):
        if self.__csr_mode:
            self.train_mode=False
        else:
            self.train_mode = mode
    def eval(self, mode = True):
        if self.__csr_mode:
            self.train_mode=False
        else:
            self.train_mode = not mode



def _pruning_hook(layer: SparseLayer, _, __):
    layer.prune_smallest_values()

 # Funkcja globalna służąca do zmniejszania wartości modelu (potrzebuje przykładowych wartości wejściowych, tzn. "dummy input")
 # "Opakowanie" klasy Pruner
def prune_sparse_model(model: nn.Module, dummy_input: torch.Tensor):
    pruner = Pruner(model)
    pruner(dummy_input)
    pruner.remove_hooks()
    model.zero_grad()

 # Funkcja ustawiająca globalną wartość paramtru k w warstwach rzadkich modelu
def set_global_k(model: nn.Module, k: float):
    for child in model.children():
        if type(child) is SparseLayer:
            child.set_k(k)

 # Klasa zmniejszająca wielkość warstw rzadkich w zadanym jako argument modelu
class Pruner(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.handles = self._register_hooks_recursive_to_sparse(self.model, _pruning_hook)

    def forward(self, input_vals):
        return self.model(input_vals)

    def _register_hooks_recursive_to_sparse(self, model, hook):
        handles = []
        for i, module in model.named_children():
            if list(module.children()):
                handles.extend(self._register_hooks_recursive(module, hook))
            if type(module) == SparseLayer:
                handle = module.register_forward_hook(_pruning_hook)
                handles.append(handle)
        return handles

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

 # Zwraca losowo zainicjalizowaną warstwę rzadką w reprezentacji COO
def new_random_basic_coo(in_features, out_features, bias=True):
    return SparseLayer(in_features, out_features, bias=bias)

 # Zwraca losowo zainicjalizowaną warstwę rzadką w reprezentacji CSR
def new_random_basic_csr(in_features, out_features, bias=True):
    return SparseLayer(in_features, out_features, bias=bias, csr_mode=True, train_mode=False)
