[<img src="http://www.opennn.net/images/deep_neural_network.png" width="500px" alt="logo" />](https://github.com/gjeusel/NCG013_redes_neurais_sem_peso)

# NCG013 Weightless Neural Network : [WiSARD](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2009-6.pdf)

----

# First Project : Hand-Written digits
----
## Purpose : Classify hand-written digits with WiSARD

Database :
- From [MNIST] (https://www.nist.gov) : National Institute of
Standards and Technology
- [Download] (http://yann.lecun.com/exdb/mnist/)

Switch to jpg database with [mnist-to-jpg.py](https://gist.github.com/ischlag/41d15424e7989b936c1609b53edd1390)

Python Library Used :
- [TensorFlow](https://www.tensorflow.org/) required by mnist-to-jpg.py
- [PIL](https://pypi.python.org/pypi/PIL)
- [PyWANN](https://github.com/firmino/PyWANN)

## Usage :
- git clone https://github.com/gjeusel/NCG013_redes_neurais_sem_peso.git
- cd NCG013_redes_neurais_sem_peso/
- python mnist-to-jpg.py
- python redes_neurais_sem_peso.py --limit_train_set 900 --limit_test_set 100

- python redes_neurais_sem_peso.py --help for more infos

----

# Final Project : Energy efficiency of buildings
----
## Purpose : Classify building's energy load according their geometry.

[<img src="http://www.toutsurlisolation.com/var/toutsurlisolation/storage/images/media/images/garder-la-route-avec-le-label-effinergie3/3049-1-fre-FR/Garder-la-route-avec-le-label-Effinergie.png" width="500px" alt="logo" />](https://fr.wikipedia.org/wiki/Diagnostic_de_performance_%C3%A9nerg%C3%A9tique)

Dataset : energy_efficiency.csv

|Mathematical representation | Input or output variable | Number of possible values | Unit |
|---|---|---|---|
|X1 | Relative Compactness | 12 | None |
|X2 | Surface Area | 12 | m²|
|X3 | Wall Area | 7 | m²|
|X4 | Roof Area | 4 | m²|
|X5 | Overall Height | 2 | m|
|X7 | Glazing Area | 4 | m²|
|X8 | Glazing Area Distribution | 6 | None |
|y | Heating Load + Cooling Load | 636 | Unknown |

## Usage :
- git clone https://github.com/gjeusel/NCG013_redes_neurais_sem_peso.git
- cd NCG013_redes_neurais_sem_peso/
- python trabalho_final.py

- python redes_neurais_sem_peso.py --help for more infos
