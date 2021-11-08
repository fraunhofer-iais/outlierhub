import io
import os

import numpy as np
from PIL import Image, ImageFile
import h5py

from data_stack.dataset.iterator import DatasetIterator
from data_stack.io.resources import StreamedResource


class ODRIterator(DatasetIterator):

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __init__(self, data_stream: StreamedResource, split_name: str):
        self.h5py_file = h5py.File(data_stream, "r")
        self.sample_groups = self.h5py_file['samples']
        self.targets_datasets = self.h5py_file['targets']

    def __len__(self):
        return len(self.sample_groups)

    def __getitem__(self, index: int):
        """
        Returns the sample and target of the dataset at given index position.
        @param index: index within dataset
        @return: sample, target, tag
        """
        if len(self) > index:
            sample_name_left, sample_name_right = self.sample_groups[str(index)]

            sample_left = self.sample_groups[str(index)][sample_name_left]
            sample_right = self.sample_groups[str(index)][sample_name_right]

            sample_left = np.array(sample_left)
            sample_right = np.array(sample_right)

            sample_left = io.BytesIO(sample_left)
            sample_right = io.BytesIO(sample_right)

            sample_left = Image.open(sample_left)
            sample_right = Image.open(sample_right)

            sample_both = (sample_left, sample_right)

            print(type(self.targets_datasets[str(index)]))
            target = np.asarray(self.targets_datasets[str(index)])

            return sample_both, target, target
        else:
            raise IndexError('index out of range')