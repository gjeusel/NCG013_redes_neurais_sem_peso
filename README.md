[<img src="http://www.opennn.net/images/deep_neural_network.png" width="500px" alt="logo" />](https://github.com/gjeusel/NCG013_redes_neurais_sem_peso)

# NCG013 Weightless Neural Network : [WiSARD](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2009-6.pdf)

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
