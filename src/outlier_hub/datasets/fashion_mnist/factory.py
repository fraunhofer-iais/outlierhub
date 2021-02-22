#!/usr/bin/env python3

import os
from data_stack.io.storage_connectors import StorageConnector, FileStorageConnector
from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from typing import Tuple, Dict, Any
from data_stack.dataset.meta import IteratorMeta
import torchvision
import tempfile
from outlier_hub.datasets.fashion_mnist.iterator import FashionMNISTIterator
from outlier_hub.datasets.fashion_mnist.preprocessor import FashionMNISTPreprocessor
from data_stack.io.resources import ResourceFactory


class FashionMNISTFactory(BaseDatasetFactory):

    def __init__(self, storage_connector: StorageConnector):
        self.raw_path = "fashion_mnist/raw/"
        self.preprocessed_path = "fashion_mnist/preprocessed/"
        self.splits = ["train", "test"]

        super().__init__(storage_connector)

    def check_exists(self, split: str) -> bool:
        # TODO come up with a better check!
        sample_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="samples.pt")
        return self.storage_connector.has_resource(sample_identifier)

    def _get_resource_id(self, data_type: str,  split: str, element: str) -> str:
        return os.path.join("fashion_mnist", data_type, split, element)

    def _retrieve_raw(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # download train split
            torchvision.datasets.FashionMNIST(root=tmpdirname, download=True, train=True)
            src = os.path.join(tmpdirname, "FashionMNIST/processed/training.pt")
            identifier = self._get_resource_id(data_type="raw", split="train", element="samples.pt")
            with open(src, "rb") as fd:
                resource = ResourceFactory.get_resource(identifier=identifier, file_like_object=fd)
                self.storage_connector.set_resource(identifier=identifier, resource=resource)

            # download test split
            torchvision.datasets.FashionMNIST(root=tmpdirname, download=True, train=False)
            src = os.path.join(tmpdirname, "FashionMNIST/processed/test.pt")
            identifier = self._get_resource_id(data_type="raw", split="test", element="samples.pt")
            with open(src, "rb") as fd:
                resource = ResourceFactory.get_resource(identifier=identifier, file_like_object=fd)
                self.storage_connector.set_resource(identifier=identifier, resource=resource)

    def _prepare_split(self):
        preprocessor = FashionMNISTPreprocessor(self.storage_connector)
        for split in self.splits:
            preprocessed_samples_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="samples.pt")
            preprocessed_targets_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="targets.pt")
            raw_identifier = self._get_resource_id(data_type="raw", split=split, element="samples.pt")
            preprocessor.preprocess(raw_identifier=raw_identifier,
                                    preprocessed_samples_identifier=preprocessed_samples_identifier,
                                    preprocessed_targets_identifier=preprocessed_targets_identifier)

    def _get_iterator(self, split: str) -> DatasetIteratorIF:
        if not self.check_exists(split):
            self._retrieve_raw()
            self._prepare_split()

        sample_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="samples.pt")
        target_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="targets.pt")
        sample_resource = self.storage_connector.get_resource(identifier=sample_identifier)
        target_resource = self.storage_connector.get_resource(identifier=target_identifier)
        return FashionMNISTIterator(sample_resource, target_resource)

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> Tuple[DatasetIteratorIF, IteratorMeta]:
        return self._get_iterator(**config), IteratorMeta(sample_pos=0, target_pos=1, tag_pos=2)


if __name__ == "__main__":
    import data_stack
    from matplotlib import pyplot as plt

    data_stack_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(data_stack.__file__))))
    example_file_storage_path = os.path.join(data_stack_root, "example_file_storage")
    storage_connector = FileStorageConnector(root_path=example_file_storage_path)

    factory = FashionMNISTFactory(storage_connector)
    factory.get_dataset_iterator(config={"split": "train"})
    iterator, _ = factory.get_dataset_iterator(config={"split": "train"})
    img, target, tag = iterator[0]
    plt.imshow(img)
    plt.show()
