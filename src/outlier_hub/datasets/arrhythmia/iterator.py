from data_stack.io.resources import StreamedTextResource
import pandas as pd
import torch
from data_stack.dataset.iterator import DatasetIterator


class ArrhythmiaIterator(DatasetIterator):

    def __init__(self, samples_stream: StreamedTextResource, targets_stream: StreamedTextResource):
        self.samples = pd.read_csv(samples_stream)
        self.targets = pd.read_csv(targets_stream)
        samples_stream.close()
        targets_stream.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        """ Returns the sample and target of the dataset at given index position.
        :param index: index within dataset
        :return: sample, target, tag
        """
        sample_tensor = torch.FloatTensor(self.samples.iloc[index].to_numpy())
        target = int(self.targets.iloc[index])
        return sample_tensor, target, target
