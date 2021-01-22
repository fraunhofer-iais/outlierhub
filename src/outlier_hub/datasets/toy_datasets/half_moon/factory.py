#!/usr/bin/env python3

from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.dataset.meta import MetaFactory
from outlier_hub.datasets.toy_datasets.half_moon.iterator import HalfMoonIterator
from typing import List, Dict, Any


class HalfMoonFactory(BaseDatasetFactory):
    """Builds a half moon dataset.
    """

    def __init__(self):
        super().__init__()

    def _get_iterator(self, split: str, noise_std: float, num_samples: List[int], seed: int = 1, translation: List[int] = None):
        meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return HalfMoonIterator(seed, noise_std, num_samples, translation), meta

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> DatasetIteratorIF:
        return self._get_iterator(**config)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    config = {"split": "train", "seed": 1, "noise_std": 0.1, "num_samples": [200, 200], "translation": [5, 5]}
    factory = HalfMoonFactory()
    iterator, meta = factory.get_dataset_iterator(config)

    samples, targets = zip(*[(s, t) for s, t, _ in iterator])

    sample_tensor = torch.stack(samples)
    moon_1 = sample_tensor[torch.IntTensor(targets) == 0]
    moon_2 = sample_tensor[torch.IntTensor(targets) == 1]

    plt.scatter(*list(zip(*moon_1)), color='red', s=1)
    plt.scatter(*list(zip(*moon_2)), color='blue', s=1)
    plt.show()
