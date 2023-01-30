[English below]

# Pakiet ActuallySparse
**Autorzy: Kacper Grzymkowski, Jakub Fołtyn**  

Pakiet do tworzenia faktycznie rzadkich sieci neuronowych w PyTorch.

## Opis
Projekt związany z pracą inżynierską pt. "*Zastosowanie metod redukcji wielkości modelu sieci neuronowej podczas procesu uczenia*". Obejmuje implementację rzadkiej warstwy liniowej, będącej rozszerzeniem warstw liniowych sieci neuronowej z biblioteki PyTorch, a także moduł konwersji, pozwalający na swobodne przekształcenia między warstwami (gęsta &rarr; rzadka oraz rzadka &rarr; gęsta).  

## Struktura folderów i plików:
* `actuallysparse/`
  - Zawiera implementacje:
    - konwertera (*converter.py*).
    - warstwy rzadkiej (*layers.py*).
* `tests/`
  - Zawiera testy działania związane odpowiednio z modułami:
    - konwertera (*test_converter.py*).
    - warstwy rzadkiej (*test_layers.py*).
    - całościowego modelu (*test_model.py*).
* `models/`
  - Zawiera skrypty tworzące i uczące modele sieci neuronowych, jak i testy porównawcze działania modeli zwykłych oraz pomniejszonych. Poszczególne pliki zawierają następujące funkcjonalności:
    - *training_loop.ipynb* - porównanie pętli dotrenowującej model (z jednoczesnym zmniejszaniem jego rozmiaru) zaimplementowanej przez autorów oraz utworzonej na podstawie funkcjonalności z biblioteki NNI.
    - *generate_baselines.ipynb* - utworzenie i wytrenowanie podstawowych modeli o różnych stopniach rzadkości, służących jako "baza porównawcza" dla dalszych analiz.
    - *memory.ipynb* - analizy zajętości pamięciowej modeli poddanych zmniejszeniu.
    - *pretrained.py* - funkcje zawierające architekrurę gotowego modelu wykorzystywanego do analiz, tzn. **VGG11_bn**.

  
  
[English]

# ActuallySparse Package
**Authors: Kacper Grzymkowski, Jakub Fołtyn**  

Package for creating actually sparse neural networks in PyTorch.

## Description
This project is a part of Bachelor of Engineering thesis titled: "*Zastosowanie metod redukcji wielkości modelu sieci neuronowej podczas procesu uczenia*".
This package contains implementation of a sparse linear neural network layer which is an extension of the standard PyTorch Linear layer, as well as a conversion module, which allows for easy conversions between different representations (like dense &rarr; sparse and sparse &rarr; dense). 

## Example usage
### Convert layers
The `actuallysparse.converter` module contains a simple API to convert a single compatible layer (torch Linear, actuallysparse SparseLayer) into a desired format via the `convert(layer, conversion_target)` function. Currently implemented conversion targets are `dense`, `coo` and `csr`.
```
>>> from actuallysparse.converter import convert, convert_model
>>> from torch import nn
>>> fc1 = nn.Linear(4, 4)
>>> fc1_sparse = convert(fc1, "coo")
>>> print(fc1)
Linear(in_features=4, out_features=4, bias=True)
>>> print(fc1_sparse)
SparseLayer(in_features=4, out_features=4, bias=True, csr_mode=False, k=0.05)
```
An entire model can be recursively converted using the `convert_model(model, layer_type_to_convert, conversion_target)` shorthand:
```
>>> classifier = nn.Sequential(
...     nn.Linear(16, 16),
...     nn.ReLU(),
...     nn.Linear(16, 3)
... )
>>> classifier_sparse = convert_model(
...     classifier,
...     nn.Linear, # layer type to convert
...     "coo"
... )
>>> print(classifier)
Sequential(
  (0): Linear(in_features=16, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=3, bias=True)
)
>>> print(classifier_sparse)
Sequential(
  (0): SparseLayer(in_features=16, out_features=16, bias=True, csr_mode=False, k=0.05)
  (1): ReLU()
  (2): SparseLayer(in_features=16, out_features=3, bias=True, csr_mode=False, k=0.05)
)
```
Note that any zero values in the weight matrix get automatically reduced during the conversion process - no need to coalesce weights.

### Layers
SparseLayer is in man ways an extension of Linear layer from Pytorch, and, as such, can be created in exact same way. There are, however, some additional parametres, namely:
 * train_mode - boolean flag enabling layer training (backward pass), True by default
 * csr_mmode - boolean flag switching layer weight format from COO to CSR (CSR is not compatible with training), False by default
 * k - float between 0.01 and 1, represents the percentage of parameters to be pruned, 0.05 by default
 What is more, SparseLayer implements prune_smallest_values() method, which prunes the k smallest value parameters from the layer.  
 Some usage examples:
 ```
 newSparseLayer = SparseLayer(
    in_features=3,
    out_features=3,
    bias=True,
    csr_mode=False,
    train_mode=True,
    k=0.05
)
dummy_input = torch.ones(3)
result = newSparseLayer(dummy_input)
 ```
 Pruning example:
 ```
 newSparseLayer = SparseLayer(3, 3)
 newSparseLayer.set_k(0.07)
 newSparseLayer.prune_smallest_values()
 ```
layers.py also contains prune_sparse_model() and set_global_k() methods, which can be used to efficiently prune all SparseLayers inside a model (the latter sets a global k value to all sparse layers, while the former pruns them), usage example:
```
sparseClassifier = nn.Sequential(
    SparseLayer(16, 16),
    nn.ReLU(),
    SparseLayer(16, 3)
)
set_global_k(sparseClassifier, 0.07)
dummy_input = torch.ones(16)
prune_sparse_model(sparseClassifier, dummy_input)
```
### Training mode
Due to limitations in PyTorch optimizers, it is currently not possible to perform a backwards pass on sparse matrices. 
Because of this, `SparseLayer` in training mode will convert its sparse matrix to a dense representation during a forward pass, rather than using the sparse representation.
For accurate timings and training capability, ensure `SparseLayer` is in correct mode using PyTorch eval/train methods (`.eval()`, `.train()`).

## Installation
As the package is not yet published on PyPI, please install it from this repo directly: 
```
# in repo root
pip install .
# or
conda develop .
```

## Directory structure

* `actuallysparse/`contains implementation of the converter module (`converter.py`) and the sparse layer (`layers.py`)
* `tests/` contains automated tests for above modules (`test_converter.py`, `test_layer.py`) and for integration between them with a full example model (`test_model.py`). These can be run using the `pytest` test framework.
* `models/` contains notebooks and scripts used for creating and training neural networks used in our analyses, as well as experiments designed to compare effectiveness, performance and memory usage of converted models.

## Acknowledgment

This project is partially based on [SparseLinear](https://github.com/hyeon95y/SparseLinear) and [PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10/) projects. Thanks [@hyeon95y](https://github.com/hyeon95y) and [@huyvnphan](https://github.com/huyvnphan) for your hard work.
