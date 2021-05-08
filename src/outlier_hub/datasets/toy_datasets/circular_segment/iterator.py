import torch
from data_stack.dataset.iterator import DatasetIterator
from typing import List
import numpy as np


class CircularSegmentIterator(DatasetIterator):

    def __init__(self, seed: int, class_label: int, radius: float, start_degree: float, end_degree: float,
                 num_samples: int, noise_std: int = 0, translation: List[int] = None):
        """
        Args:
            seed (int): seed for random generator initialization
            num_samples (List): List of length 2 indicating the number of samples for each half moon
        """
        self.class_label = class_label
        rng = np.random.default_rng(seed=seed)
        X_degrees = torch.FloatTensor(rng.uniform(low=[start_degree], high=[end_degree], size=(num_samples,)))
        self.X = torch.stack([radius*torch.cos(X_degrees), radius*torch.sin(X_degrees)], dim=1)
        if translation is not None:
            self.X = self.X + torch.Tensor(translation)
        # add gaussian noise
        noise = torch.Tensor(rng.normal(loc=0, scale=noise_std, size=num_samples*2)).reshape(num_samples, 2)
        self.X = self.X + noise

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index: int):
        """ Returns the sample and target of the dataset at given index position.
        :param index: index within dataset
        :return: sample, target, tag
        """
        return self.X[index], self.class_label, self.class_label
