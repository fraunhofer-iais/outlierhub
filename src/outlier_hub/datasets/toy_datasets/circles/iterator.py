import torch
from data_stack.dataset.iterator import DatasetIterator
from typing import List
from sklearn.datasets import make_circles


class CirclesIterator(DatasetIterator):

    def __init__(self, seed: int, noise_std: float, num_samples: List[int], scale_factor: float, translation: List[int] = None):
        """
        Creates a dataset iterator for the circle within circle dataset

        Args:
            seed (int): seed for random generator initialization
            noise_std (float): standard deviation of noise
            num_samples (List): List of length 2 indicating the number of samples for each half moon
            scale_factor (float): Scale factor between inner and outer circle in the range (0, 1)
        """

        self.X, self.y = make_circles(n_samples=num_samples, shuffle=True, noise=noise_std, random_state=seed, factor=scale_factor)
        self.X = torch.Tensor(self.X)
        if translation is not None:
            self.X = self.X + torch.Tensor(translation)
        self.y = torch.Tensor(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):
        """ Returns the sample and target of the dataset at given index position.
        :param index: index within dataset
        :return: sample, target, tag
        """
        return self.X[index], int(self.y[index]), int(self.y[index])
