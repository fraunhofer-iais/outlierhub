import os
import h5py
import glob
from numpy.core.fromnumeric import argmax
import torch
from data_stack.dataset.iterator import DatasetIterator
from data_stack.io.resources import StreamedResource


class HAMIterator(DatasetIterator):

    def __init__(self, data_stream: StreamedResource):
        self.h5py_file = h5py.File(data_stream, "r")
        self.samples_dataset = self.h5py_file["samples"]
        self.targets_dataset = self.h5py_file["targets"]

    def __len__(self):
        return len(self.samples_dataset)

    def __getitem__(self, index: int):
        if(len(self) > index):
            target = argmax(self.targets_dataset[index])
            return torch.from_numpy(self.samples_dataset[index]).permute(2,0,1), target, target
        else:
            raise IndexError('index out of range')