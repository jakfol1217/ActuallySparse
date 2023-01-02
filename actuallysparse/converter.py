import copy

from actuallysparse import layers
from actuallysparse.layers import SparseLayer
from torch.nn.modules import Linear
from torch.nn import Module
from torch import Tensor

 # Funkcja konwertująca warstwy
def convert(layer: Linear | SparseLayer, convert_target: str, mask: None | Tensor = None):
    extractor = match_extractor(layer)
    packager = match_packager(convert_target)

    weight, bias = extractor(layer)
    if mask is not None:
        weight = weight * mask

    return packager(weight, bias)

 # Funkcja konwertująca cały model
def convert_model(model: Module, module_to_replace: Module, target: str):
    new_model = copy.deepcopy(model)
    for i, module in new_model.named_children():
        if list(module.children()):
            setattr(new_model, i, convert_model(module, module_to_replace, target))
        if type(module) == module_to_replace:
            setattr(new_model, i, convert(module, target))
    return new_model

 # Funkcja sprawdzająca i dopasowująca warstwę, która ma zostać poddana konwersji
def match_extractor(layer):
    if isinstance(layer, Linear):
        return extract_params_dense
    elif isinstance(layer, SparseLayer) and not layer.csr_mode:
        return extract_params_sparse_coo
    elif isinstance(layer, SparseLayer) and layer.csr_mode:
        return extract_params_sparse_csr
    else:
        raise TypeError("Convert module requires layer to be Linear or SparseLayer")

 # Funkcja sprawdzająca i dopasowująca cel konwersji
def match_packager(convert_target: str):
    if convert_target == "dense":
        return package_params_dense
    elif convert_target == "coo":
        return package_params_coo
    elif convert_target == "csr":
        return package_params_csr
    else:
        raise TypeError(f"Unknown convert_target: '{convert_target}'")

 # Funkcja pozyskująca parametry warstwy gęstej
def extract_params_dense(layer: Linear):
    # Klonowanie wartości, a nie tylko referencji, żeby operacje wykonane na jednej z warstw nie miały wpływu na drugą
    return layer.weight.data.clone().detach(), layer.bias.data.clone().detach()

 # Funkcja pozyskująca parametry warstwy rzadkiej
def extract_params_sparse_coo(layer: SparseLayer):
    return layer.weight.data.to_dense(), layer.bias.data.clone().detach()


extract_params_sparse_csr = extract_params_sparse_coo

# Funkcja przypisująca parametry w warstwie gęstej
def package_params_dense(weight, bias):
    out_features, in_features = weight.size()
    converted_layer = Linear(in_features, out_features)

    converted_layer.weight.data = weight
    converted_layer.bias.data = bias

    return converted_layer

# Funkcja przypisująca parametry w warstwie rzadkiej
def package_params_sparse(weight, bias, constructor):
    out_features, in_features = weight.size()
    converted_layer = constructor(in_features, out_features)
    converted_layer.assign_new_weight(weight, bias=bias)

    return converted_layer


def package_params_coo(weight, bias):
    return package_params_sparse(weight, bias, layers.new_random_basic_coo)


def package_params_csr(weight, bias):
    return package_params_sparse(weight, bias, layers.new_random_basic_csr)
