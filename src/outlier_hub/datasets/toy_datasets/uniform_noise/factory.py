#!/usr/bin/env python3

from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.dataset.meta import MetaFactory
from outlier_hub.datasets.toy_datasets.uniform_noise.iterator import UniformNoiseIterator
from typing import List, Tuple, Dict, Any


class UniformNoiseFactory(BaseDatasetFactory):
    """Builds a half moon dataset.
    """

    def __init__(self):
        super().__init__()

    def _get_iterator(self, split: str, num_samples: List[int], classes: List[int], hypercube: List[Tuple[int, int]], seed: int = 1):
        meta = MetaFactory.get_iterator_meta(
            sample_pos=0, target_pos=1, tag_pos=2)
        return UniformNoiseIterator(seed=seed, num_samples=num_samples, classes=classes, hypercube=hypercube), meta

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> DatasetIteratorIF:
        return self._get_iterator(**config)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    classes = [0, 1]
    hypercube = [(-1, 1), (3, 6)]
    config = {"seed": 1, "classes": classes, "num_samples": [
        2000, 2000], "hypercube": hypercube, "split": "full"}
    factory = UniformNoiseFactory()
    iterator, meta = factory.get_dataset_iterator(config)

    samples, targets = zip(*[(s, t) for s, t, _ in iterator])

    sample_tensor = torch.stack(samples)
    class_0_samples = sample_tensor[torch.IntTensor(targets) == 0]
    class_1_samples = sample_tensor[torch.IntTensor(targets) == 1]

    plt.scatter(*list(zip(*class_0_samples)), color='red', s=1)
    plt.scatter(*list(zip(*class_1_samples)), color='blue', s=1)
    plt.show()
