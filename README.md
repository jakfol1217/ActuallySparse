# PyTorch ActuallySparse Module
* **Autorzy: Kacper Grzymkowski, Jakub Fołtyn**  

Projekt związany z pracą inżynierską pt. "*Zastosowanie metod redukcji wielkości modelu sieci neuronowej podczas procesu uczenia*". Obejmuje implementację rzadkiej warstwy liniowej, będącej rozszerzeniem warstw liniowych sieci neuronowej z biblioteki PyTorch, a także moduł konwersji, pozwalający na swobodne przekształcenia między warstwami (gęsta &rarr; rzadka oraz rzadka &rarr; gęsta).  

## Struktura folderów i plików:
* `actuallysparse\`
  - Zawiera implementacje:
    - konwertera (*converter.py*).
    - warstwy rzadkiej (*layers.py*).
* `tests\`
  - Zawiera testy działania związane odpowiednio z modułami:
    - konwertera (*test_converter.py*).
    - warstwy rzadkiej (*test_layers.py*).
    - całościowego modelu (*test_model.py*).
* `modules\`
  - Zawiera skrypty tworzące i uczące modele sieci neuronowych, jak i testy porównawcze działania modeli zwykłych oraz pomniejszonych. Poszczególne pliki zawierają następujące funkcjonalności:
    - *training_loop.ipynb* - porównanie pętli dotrenowującej model (z jednoczesnym zmniejszaniem jego rozmiaru) zaimplementowanej przez autorów oraz utworzonej na podstawie funkcjonalności z biblioteki NNI.
    - *generate_baselines.ipynb* - utworzenie i wytrenowanie podstawowych modeli o różnych stopniach rzadkości, służących jako "baza porównawcza" dla dalszych analiz.
    - *integrate.ipynb* -
    - *memory.ipynb* - analizy zajętości pamięciowej modeli poddanych zmniejszeniu.
    - *pretrained.py* - funkcje zawierające architekrurę gotowego modelu wykorzystywanego do analiz, tzn. **VGG11_bn**
    - *sparse_loop.ipynb* -
    - *sparsify.ipynb* -
  
