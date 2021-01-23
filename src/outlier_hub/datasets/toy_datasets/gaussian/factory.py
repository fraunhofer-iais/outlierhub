#!/usr/bin/env python3

from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.dataset.meta import MetaFactory
from outlier_hub.datasets.toy_datasets.gaussian.iterator import GaussianIterator
from typing import Dict, Any, Tuple
import numpy as np


class GaussianFactory(BaseDatasetFactory):
    """Builds a parameterizable Gaussian dataset.
    """

    def __init__(self):
        super().__init__()

    def _get_iterator(self, split: str, class_label: int, seed: int, num_samples: int, covariance: np.array, mean: Tuple[int, int]):
        meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return GaussianIterator(seed, class_label, num_samples, covariance, mean), meta

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> DatasetIteratorIF:
        return self._get_iterator(**config)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    config = {"split": "train", "seed": 1, "class_label": 1, "num_samples": 10000, "covariance": [[5, -5], [-5, 5]], "mean": (-5, 5)}
    factory = GaussianFactory()
    iterator, meta = factory.get_dataset_iterator(config)

    samples, targets = zip(*[(s, t) for s, t, _ in iterator])

    sample_tensor = torch.stack(samples)

    plt.scatter(*list(zip(*sample_tensor)), color='red', s=1)
    plt.show()
