# OutlierHub

[![CircleCI](https://circleci.com/gh/fraunhofer-iais/outlierhub.svg?style=svg)](https://circleci.com/gh/fraunhofer-iais/outlierhub)


a curated hub for outlier and anomaly datasets built on top of [datastack](https://github.com/le1nux/datastack).

Currentyly supported datasets:

* ATIS
* [ARRHYTHMIA](https://archive.ics.uci.edu/ml/datasets/arrhythmia)
* KDD
* [REUTERS](https://www.nltk.org/book/ch02.html)
* [TREC](https://cogcomp.seas.upenn.edu/Data/QA/QC/)
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
* [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)
* [Newsgroups](http://qwone.com/~jason/20Newsgroups/)

Furthermore, OutlierHub supports the following toy datasets:

* Half Moons
* Nested Circles
* Noisy x -> x^3 regression
* Uniform Noise
* Gaussian Cluster
* Circular Segement
* XOR Squares

# Install 

```bash
pip install outlier-hub
```

For the latest version, clone or download the repository and `cd` into OutlierHub's root folder and run

```bash
pip install src/
```



# Usage

Each dataset has a factory that can be instantiated with a `StorageConnector` providing the IO operations. 
After factory instantiation, the factory is able to create a dataset iterator when calling its member method `get_dataset_iterator(config={...})`  

**Copyright 2020 Fraunhofer IAIS**

For license please see: https://github.com/fraunhofer-iais/outlierhub/blob/master/LICENSE
