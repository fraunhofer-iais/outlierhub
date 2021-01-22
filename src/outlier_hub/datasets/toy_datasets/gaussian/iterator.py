import torch
from data_stack.dataset.iterator import DatasetIterator
from typing import Tuple
import numpy as np


class GaussianIterator(DatasetIterator):

    def __init__(self, seed: int, class_label: int, num_samples: int, covariance: np.array, mean: Tuple[int, int]):
        """Builds a Gaussian dataset iterator.

        Args:
            seed (int): Random generator seed
            class_label (int): class label for all samples
            num_samples (int): number of samples within iterator
            covariance (np.array): Covariance matrix of Gaussian
            mean (Tuple[int, int]): Mean of Gaussian
        """
        rng = np.random.default_rng(seed=seed)
        self.X = torch.FloatTensor(rng.multivariate_normal(mean=mean, cov=covariance, size=num_samples))

        self.X = torch.Tensor(self.X)
        self.y = torch.ones(num_samples) * class_label

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):
        """ Returns the sample and target of the dataset at given index position.
        :param index: index within dataset
        :return: sample, target, tag
        """
        return self.X[index], int(self.y[index]), int(self.y[index])
