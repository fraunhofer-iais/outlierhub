import torch
from data_stack.dataset.iterator import DatasetIterator
from typing import List
import numpy as np


class NoisyXCubedIterator(DatasetIterator):

    def __init__(self, seed: int, noise_std: float, interval: List[float], num_samples: int):
        """
        Creates a dataset iterator over the function x -> x^3 + e, where e ~ N(0, noise_std^2).

        Args:
            seed (int): seed for random generator initialization
            noise_std (float): standard deviation of noise
            interval (List[float]): interval the samples are drawn from uniformly
            num_samples (int): number of samples
        """
        rng = np.random.default_rng(seed=seed)
        self.x_vals = torch.FloatTensor(rng.uniform(*interval, num_samples))
        perturbations = torch.FloatTensor(rng.normal(0, noise_std, num_samples))
        self.y_vals = self.x_vals**3 + perturbations

    def __len__(self):
        return len(self.y_vals)

    def __getitem__(self, index: int):
        """ Returns the sample and target of the dataset at given index position.
        :param index: index within dataset
        :return: sample, target, tag
        """
        return self.x_vals[index], self.y_vals[index], self.y_vals[index]
