import os

import h5py
import torch
from data_stack.dataset.iterator import DatasetIterator
from data_stack.io.resources import StreamedResource


class HAMIterator(DatasetIterator):

    def __init__(self, data_stream: StreamedResource, split_name: str):
        self.h5py_file = h5py.File(data_stream, "r")
        self.samples_dataset = self.h5py_file[os.path.join(split_name, "samples")]
        self.targets_dataset = self.h5py_file[os.path.join(split_name, "targets")]

    def __len__(self):
        return len(self.samples_dataset)

    def __getitem__(self, index: int):
        """
        Returns the sample and target of the dataset at given index position.
        @param index: index within dataset
        @return: sample, target, tag
        """
        if(len(self) > index):
            target = self.targets_dataset[index]
            return torch.from_numpy(self.samples_dataset[index]), target, target
        else:
            raise IndexError('index out of range')