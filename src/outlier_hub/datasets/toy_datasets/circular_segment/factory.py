#!/usr/bin/env python3

from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.dataset.meta import MetaFactory
from outlier_hub.datasets.toy_datasets.circular_segment.iterator import CircularSegmentIterator
from typing import List, Dict, Any


class CircularSegmentFactory(BaseDatasetFactory):
    """Builds a circular segment dataset.
    """

    def __init__(self):
        super().__init__()

    def _get_iterator(self, split: str, class_label: int, radius: float, start_degree: float, end_degree: float,
                      num_samples: int, seed: int = 1, translation: List[int] = None, noise_std: int = 0):
        meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return CircularSegmentIterator(seed, class_label, radius, start_degree, end_degree, num_samples, noise_std, translation), meta

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> DatasetIteratorIF:
        return self._get_iterator(**config)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    config = {"split": "train", "seed": 1, "class_label": 1, "radius": 10,
              "start_degree": 0, "end_degree": 4.71, "num_samples": 3000, "translation": [5, 5], "noise_std": 1}
    factory = CircularSegmentFactory()
    iterator, meta = factory.get_dataset_iterator(config)

    samples, targets = zip(*[(s, t) for s, t, _ in iterator])

    sample_tensor = torch.stack(samples)

    plt.scatter(*list(zip(*sample_tensor)), color='red', s=1)
    plt.show()
