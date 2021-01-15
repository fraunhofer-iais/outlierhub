#!/usr/bin/env python3

from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.dataset.meta import MetaFactory
from outlier_hub.datasets.toy_datasets.noisy_x_cubed_regression.iterator import NoisyXCubedIterator
from typing import List


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

    def __init__(self, seed: int = 1, noise_std: float = 3, interval: List[float] = None, num_samples: int = 20):
        super().__init__()
        self.seed = seed
        self.noise_std = noise_std
        self.num_samples = num_samples
        self.interval = [-4, 4] if interval is None else interval

    def _get_iterator(self):
        meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return NoisyXCubedIterator(self.seed, self.noise_std, self.interval, self.num_samples), meta

    def get_dataset_iterator(self, split: str = None) -> DatasetIteratorIF:
        return self._get_iterator()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    factory = NoisyXCubedFactory(seed=1)
    iterator, meta = factory.get_dataset_iterator()

    samples, targets = zip(*[(s, t) for s, t, _ in iterator])

    plt.plot(samples, targets, 'o', color='black')
    # plot true function
    x_true, y_true = zip(*[(i, i**3) for i in np.linspace(-6, 6, 100)])
    plt.plot(x_true, y_true, '-')

    plt.show()
