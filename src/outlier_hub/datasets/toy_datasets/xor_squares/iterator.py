import torch
from data_stack.dataset.iterator import DatasetIterator
from typing import List
import numpy as np


class XORSquaresIterator(DatasetIterator):

    def __init__(self, seed: int, length: float, num_samples: List[int], translation: List[int] = None):
        """
        Args:
            seed (int): seed for random generator initialization
            length (float):
            num_samples (List): List of length 2 indicating the number of samples for each half moon
        """
        rng = np.random.default_rng(seed=seed)
        square_length = length/2
        # -length/2 0    length/2
        #   x-------|--------x
        #   |       |        |
        #   |  0 0  |   0 1  |
        # 0 |----------------x
        #   |  1 0  |   1 1  |
        #   |       |        |
        #   x-------|--------x

        self.X_0_0 = torch.FloatTensor(rng.uniform(low=[-square_length, 0], high=[0, square_length], size=(num_samples[0], 2)))
        self.X_0_1 = torch.FloatTensor(rng.uniform(low=[0, 0], high=[square_length, square_length], size=(num_samples[1], 2)))
        self.X_1_0 = torch.FloatTensor(rng.uniform(low=[-square_length, -square_length], high=[0, 0], size=(num_samples[2], 2)))
        self.X_1_1 = torch.FloatTensor(rng.uniform(low=[0, -square_length], high=[square_length, 0], size=(num_samples[3], 2)))
        self.X = torch.cat([self.X_0_0, self.X_0_1, self.X_1_0, self.X_1_1])
        if translation is not None:
            self.X = self.X + torch.Tensor(translation)

        self.y_0_0 = torch.Tensor([1]*num_samples[0])
        self.y_0_1 = torch.Tensor([0]*num_samples[1])
        self.y_1_0 = torch.Tensor([0]*num_samples[2])
        self.y_1_1 = torch.Tensor([1]*num_samples[3])
        self.y = torch.cat([self.y_0_0, self.y_0_1, self.y_1_0, self.y_1_1])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):
        """ Returns the sample and target of the dataset at given index position.
        :param index: index within dataset
        :return: sample, target, tag
        """
        return self.X[index], int(self.y[index]), int(self.y[index])
