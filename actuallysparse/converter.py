from actuallysparse import layers
from actuallysparse.layers import SparseLayer
from torch.nn.modules import Linear
from torch import Tensor


def convert(layer: Linear | SparseLayer, convert_target: str, mask: None | Tensor = None):
    extractor = match_extractor(layer)
    packager = match_packager(convert_target)

    weight, bias = extractor(layer)
    if mask is not None:
        weight = weight * mask

    return packager(weight, bias)


def match_extractor(layer):
    if isinstance(layer, Linear):
        return extract_params_dense
    elif isinstance(layer, SparseLayer) and not layer.csr_mode:
        return extract_params_sparse_coo
    elif isinstance(layer, SparseLayer) and layer.csr_mode:
        return extract_params_sparse_csr
    else:
        raise TypeError("Convert module requires layer to be Linear or SparseLayer")


def match_packager(convert_target: str):
    if convert_target == "dense":
        return package_params_dense
    elif convert_target == "coo":
        return package_params_coo
    elif convert_target == "csr":
        return package_params_csr
    else:
        raise TypeError(f"Unknown convert_target: '{convert_target}'")


def extract_params_dense(layer: Linear):
    # Klonowanie wartości, a nie tylko referencji, żeby operacje wykonane na jednej z warstw nie miały wpływu na drugą
    return layer.weight.data.clone().detach(), layer.bias.data.clone().detach()


def extract_params_sparse_coo(layer: SparseLayer):
    return layer.weight.data.to_dense(), layer.bias.data.clone().detach()


extract_params_sparse_csr = extract_params_sparse_coo


def package_params_dense(weight, bias):
    out_features, in_features = weight.size()
    converted_layer = Linear(in_features, out_features)

    converted_layer.weight.data = weight
    converted_layer.bias.data = bias

    return converted_layer


def package_params_sparse(weight, bias, constructor):
    out_features, in_features = weight.size()
    converted_layer = constructor(in_features, out_features)
    converted_layer.assign_new_weight(weight.t(), bias=bias)

    return converted_layer


def package_params_coo(weight, bias):
    return package_params_sparse(weight, bias, layers.new_random_basic_coo)


def package_params_csr(weight, bias):
    return package_params_sparse(weight, bias, layers.new_random_basic_csr)
