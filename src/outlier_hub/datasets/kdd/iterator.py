from data_stack.io.resources import StreamedTextResource
import pandas as pd
from data_stack.dataset.iterator import DatasetIterator


class KDDIterator(DatasetIterator):

    def __init__(self, samples_stream: StreamedTextResource):
        df = pd.read_csv(samples_stream)
        samples_stream.close()
        self.samples = df.loc[:, df.columns != "target"]
        self.targets = df["target"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        """ Returns the sample and target of the dataset at given index position.
        :param index: index within dataset
        :return: sample, target, tag
        """
        sample_tensor = self.samples.iloc[index].to_numpy()
        target = self.targets.iloc[index]
        return sample_tensor, target, target
