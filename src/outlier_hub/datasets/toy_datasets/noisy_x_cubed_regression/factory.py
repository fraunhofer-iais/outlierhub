#!/usr/bin/env python3

from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.dataset.meta import MetaFactory
from outlier_hub.datasets.toy_datasets.noisy_x_cubed_regression.iterator import NoisyXCubedIterator
from typing import List, Dict, Any


class NoisyXCubedFactory(BaseDatasetFactory):
    """Generates dataset as defind by

    @inproceedings{hernandez2015probabilistic,
    title={Probabilistic backpropagation for scalable learning of bayesian neural networks},
    author={Hern{\'a}ndez-Lobato, Jos{\'e} Miguel and Adams, Ryan},
    booktitle={International Conference on Machine Learning},
    pages={1861--1869},
    year={2015}
    }

    Args:
        BaseDatasetFactory ([type]): [description]
    """

    def __init__(self):
        super().__init__()

    def _get_iterator(self, noise_std: float, interval: List[float], num_samples: int, seed: int = 1):
        meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return NoisyXCubedIterator(seed, noise_std, interval, num_samples), meta

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> DatasetIteratorIF:
        return self._get_iterator(**config)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    interval = [-10, 10]
    config = {"seed": 1, "noise_std": 3, "interval": interval, "num_samples": 30}
    factory = NoisyXCubedFactory()
    iterator, meta = factory.get_dataset_iterator(config)

    samples, targets = zip(*[(s, t) for s, t, _ in iterator])

    plt.plot(samples, targets, 'o', color='black')
    # plot true function
    x_true, y_true = zip(*[(i, i**3) for i in np.linspace(*(np.array(interval) + [-2, 2]), 100)])
    plt.plot(x_true, y_true, '-')

    plt.show()
