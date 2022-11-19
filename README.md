# PyTorch ActuallySparse Module
* **Autorzy: Kacper Grzymkowski, Jakub Fołtyn**  

Projekt związany z pracą inżynierską pt. "*Zastosowanie metod redukcji wielkości modelu sieci neuronowej podczas procesu uczenia*". Obejmuje implementację rzadkiej warstwy liniowej, będącej rozszerzeniem warstw liniowych sieci neuronowej z biblioteki PyTorch, a także moduł konwersji, pozwalający na swobodne przekształcenia między warstwami (gęsta &rarr; rzadka oraz rzadka &rarr; gęsta).  

## Struktura folderów:
* `actuallysparse`
  - Zawiera implementacje konwertera (*converter.py*) oraz warstwy rzadkiej (*layers.py*).
* `tests`
  - Zawiera testy związane odpowiednio z modułami:
    - konwertera (*test_converter.py*).
    - warstwy rzadkiej (*test_layers.py*).
