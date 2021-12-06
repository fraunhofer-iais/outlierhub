import io
import os

import h5py
import torch
from data_stack.dataset.iterator import DatasetIterator
from data_stack.io.resources import StreamedResource
import numpy as np
from PIL import Image, ImageFile


class HAMIterator(DatasetIterator):

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __init__(self, data_stream: StreamedResource, split_name: str):
        self.h5py_file = h5py.File(data_stream, "r")
        self.samples_datasets = self.h5py_file["samples"]
        self.targets_datasets = self.h5py_file["targets"]

    def __len__(self):
        return len(self.samples_datasets)

    def __getitem__(self, index: int):
        """
        Returns the sample and target of the dataset at given index position.
        @param index: index within dataset
        @return: sample, target, tag
        """
        if len(self) > index:
            sample_np = np.array(self.samples_datasets[str(index)])
            sample_bytes = io.BytesIO(sample_np)
            sample = Image.open(sample_bytes)
            target = np.asarray(self.targets_datasets[str(index)])

            return sample, target, target
        else:
            raise IndexError('index out of range')