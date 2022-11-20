# PyTorch ActuallySparse Module
* **Autorzy: Kacper Grzymkowski, Jakub Fołtyn**  

Projekt związany z pracą inżynierską pt. "*Zastosowanie metod redukcji wielkości modelu sieci neuronowej podczas procesu uczenia*". Obejmuje implementację rzadkiej warstwy liniowej, będącej rozszerzeniem warstw liniowych sieci neuronowej z biblioteki PyTorch, a także moduł konwersji, pozwalający na swobodne przekształcenia między warstwami (gęsta &rarr; rzadka oraz rzadka &rarr; gęsta).  

## Struktura folderów i plików:
* `\actuallysparse\`
  - Zawiera implementacje:
    - konwertera (*converter.py*).
    - warstwy rzadkiej (*layers.py*).
* `\tests\`
  - Zawiera testy związane odpowiednio z modułami:
    - konwertera (*test_converter.py*).
    - warstwy rzadkiej (*test_layers.py*).
    - całościowego modelu (*test_model.py*).
* `\sparsify.ipynb` - zawiera implementację modułu przygotowującego korzystającego z biblioteki *NNI*
