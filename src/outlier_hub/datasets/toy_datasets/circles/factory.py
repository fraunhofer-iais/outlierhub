#!/usr/bin/env python3

from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.dataset.meta import MetaFactory
from outlier_hub.datasets.toy_datasets.circles.iterator import CirclesIterator
from typing import List


class CirclesFactory(BaseDatasetFactory):
    """Builds a half moon dataset.
    """

    def __init__(self, seed: int = 1, scale_factor: float = 0.8, noise_std: float = None, num_samples: List[int] = None):
        super().__init__()
        self.seed = seed
        self.noise_std = noise_std
        self.num_samples = num_samples if num_samples is not None else [100, 100]
        self.scale_factor = scale_factor

    def _get_iterator(self):
        meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return CirclesIterator(self.seed, self.noise_std, self.num_samples, self.scale_factor), meta

    def get_dataset_iterator(self, split: str = None) -> DatasetIteratorIF:
        return self._get_iterator()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    factory = CirclesFactory(seed=1, noise_std=0.1, num_samples=[2000, 2000], scale_factor=0.5)
    iterator, meta = factory.get_dataset_iterator()

    samples, targets = zip(*[(s, t) for s, t, _ in iterator])

    sample_tensor = torch.stack(samples)
    moon_1 = sample_tensor[torch.IntTensor(targets) == 0]
    moon_2 = sample_tensor[torch.IntTensor(targets) == 1]

    plt.scatter(*list(zip(*moon_1)), color='red', s=1)
    plt.scatter(*list(zip(*moon_2)), color='blue', s=1)
    plt.show()
