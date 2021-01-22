import torch
from data_stack.dataset.iterator import DatasetIterator
from typing import List
from sklearn.datasets import make_moons


class HalfMoonIterator(DatasetIterator):

    def __init__(self, seed: int, noise_std: float, num_samples: List[int], translation: List[int] = None):
        """
        Creates a dataset iterator over the function x -> x^3 + e, where e ~ N(0, noise_std^2).

        Args:
            seed (int): seed for random generator initialization
            noise_std (float): standard deviation of noise
            num_samples (List): List of length 2 indicating the number of samples for each half moon
        """
        self.X, self.y = make_moons(n_samples=num_samples, shuffle=True, noise=noise_std, random_state=seed)
        self.X = torch.Tensor(self.X)
        if translation is not None:
            self.X = self.X + torch.Tensor(translation)
        self.y = torch.IntTensor(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):
        """ Returns the sample and target of the dataset at given index position.
        :param index: index within dataset
        :return: sample, target, tag
        """
        return self.X[index], int(self.y[index]), int(self.y[index])
