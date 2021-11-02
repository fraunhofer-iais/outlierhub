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
        if len(self) > index:
            # prepare sample for output
            # first get the numpy which inherits a bytetype content
            sample_np = (self.samples_dataset[index]).tobytes()
            # transform it to bytes and open it with PIL
            sample_bytes = io.BytesIO(sample_np)
            sample = Image.open(sample_bytes)

            target = self.targets_dataset[index]

            return sample, target, target
        else:
            raise IndexError('index out of range')