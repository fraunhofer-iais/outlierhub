import torch
from data_stack.dataset.iterator import DatasetIterator
from typing import List, Tuple
import numpy as np


class UniformNoiseIterator(DatasetIterator):

    def __init__(self, num_samples: List[int], classes: List[int], hypercube: List[Tuple[int, int]], seed: int = 1):
        """
        Creates a dataset iterator for uniform noise
        Args:
         ...
        """
        rng = np.random.default_rng(seed=seed)
        min_vals = list(zip(*hypercube))[0]
        max_vals = list(zip(*hypercube))[1]
        X_list = []
        y_list = []
        for i in range(len(classes)):
            X_list.append(rng.uniform(low=min_vals, high=max_vals, size=[num_samples[i], len(hypercube)]))
            y_list.append([classes[i]]*num_samples[i])

        self.X = torch.Tensor(np.concatenate(X_list))
        self.y = torch.Tensor(np.concatenate(y_list))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):
        """ Returns the sample and target of the dataset at given index position.
        :param index: index within dataset
        :return: sample, target, tag
        """
        return self.X[index], int(self.y[index]), int(self.y[index])
