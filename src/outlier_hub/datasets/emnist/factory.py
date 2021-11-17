#!/usr/bin/env python3

import os
from data_stack.io.storage_connectors import StorageConnector, FileStorageConnector
from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from typing import Tuple, Dict, Any
from data_stack.dataset.meta import IteratorMeta
import torchvision
import tempfile
from outlier_hub.datasets.emnist.iterator import EMNISTIterator
from outlier_hub.datasets.emnist.preprocessor import EMNISTPreprocessor
from data_stack.io.resources import ResourceFactory


class EMNISTFactory(BaseDatasetFactory):

    def __init__(self, storage_connector: StorageConnector):
        self.raw_path = "emnist/raw/"
        self.preprocessed_path = "emnist/preprocessed/"
        self.splits = ["train", "test"]
        super().__init__(storage_connector)

    def check_exists(self, split: str) -> bool:
        # TODO come up with a better check!
        sample_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="samples.pt")
        return self.storage_connector.has_resource(sample_identifier)

    def _get_resource_id(self, data_type: str,  split: str, element: str) -> str:
        return os.path.join("emnist", data_type, split, element)

    def _retrieve_raw(self, split: str):
        with tempfile.TemporaryDirectory() as tmpdirname:
            split_emnist, split_inner = split.split("_")
            is_train_split = split_inner == "train"
            # download train split
            torchvision.datasets.EMNIST(root=tmpdirname, download=True, train=is_train_split, split=split_emnist)
            file_name = f"training_{split_emnist}.pt" if is_train_split else f"test_{split_emnist}.pt"
            src = os.path.join(tmpdirname, f"EMNIST/processed/{file_name}")
            identifier = self._get_resource_id(data_type="raw", split=split, element="samples.pt")
            with open(src, "rb") as fd:
                resource = ResourceFactory.get_resource(identifier=identifier, file_like_object=fd)
                self.storage_connector.set_resource(identifier=identifier, resource=resource)

    def _prepare_split(self, split: str):
        preprocessor = EMNISTPreprocessor(self.storage_connector)
        preprocessed_samples_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="samples.pt")
        preprocessed_targets_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="targets.pt")
        raw_identifier = self._get_resource_id(data_type="raw", split=split, element="samples.pt")
        preprocessor.preprocess(raw_identifier=raw_identifier,
                                preprocessed_samples_identifier=preprocessed_samples_identifier,
                                preprocessed_targets_identifier=preprocessed_targets_identifier)

    def _get_iterator(self, split: str) -> DatasetIteratorIF:
        if not self.check_exists(split):
            self._retrieve_raw(split)
            self._prepare_split(split)

        sample_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="samples.pt")
        target_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="targets.pt")
        sample_resource = self.storage_connector.get_resource(identifier=sample_identifier)
        target_resource = self.storage_connector.get_resource(identifier=target_identifier)
        return EMNISTIterator(sample_resource, target_resource)

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> Tuple[DatasetIteratorIF, IteratorMeta]:
        return self._get_iterator(**config), IteratorMeta(sample_pos=0, target_pos=1, tag_pos=2)


if __name__ == "__main__":
    import data_stack
    from matplotlib import pyplot as plt
    import string

    data_stack_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(data_stack.__file__))))
    example_file_storage_path = os.path.join(data_stack_root, "example_file_storage")
    storage_connector = FileStorageConnector(root_path=example_file_storage_path)

    factory = EMNISTFactory(storage_connector)
    alphabet = list(string.ascii_lowercase)
    iterator, _ = factory.get_dataset_iterator(config={"split": "letters_train"})
    for i in range(1):
        img, target, tag = iterator[i+20]
        print(alphabet[target])
        plt.imshow(img)
        plt.show()
    from collections import Counter
    targets = [t for _, t, _ in iterator]
    counts = dict(Counter(targets))
    print(counts)
