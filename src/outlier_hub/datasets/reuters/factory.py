#!/usr/bin/env python3

import os
from data_stack.io.storage_connectors import StorageConnector, FileStorageConnector
from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from outlier_hub.datasets.reuters.preprocessor import ReutersPreprocessor
from outlier_hub.datasets.reuters.iterator import ReutersIterator
import nltk
from data_stack.dataset.meta import MetaFactory
from typing import Dict, Any


class ReutersFactory(BaseDatasetFactory):

    def __init__(self, storage_connector: StorageConnector):
        self.raw_path = "reuters/raw/"
        self.preprocessed_path = "reuters/preprocessed/"

        super().__init__(storage_connector)

    def check_exists(self) -> bool:
        # TODO come up with a better check!
        sample_identifier = self._get_resource_id(element="reuters.hdf5")
        return self.storage_connector.has_resource(sample_identifier)

    def _get_resource_id(self, element: str) -> str:
        return os.path.join(self.preprocessed_path, element)

    def _retrieve_raw(self):
        nltk.download("reuters")

    def _prepare(self):
        dataset_identifier = self._get_resource_id(element="reuters.hdf5")
        preprocessor = ReutersPreprocessor(self.storage_connector)
        preprocessor.preprocess(dataset_identifier=dataset_identifier)

    def _get_iterator(self, split: str):
        dataset_identifier = self._get_resource_id(element="reuters.hdf5")
        dataset_resource = self.storage_connector.get_resource(identifier=dataset_identifier)
        meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return ReutersIterator(dataset_resource, split), meta

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> DatasetIteratorIF:
        if not self.check_exists():
            self._retrieve_raw()
            self._prepare()
        return self._get_iterator(**config)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        example_file_storage_path = os.path.join(root, "dataset_storage")
        storage_connector = FileStorageConnector(root_path=example_file_storage_path)

        reuters_factory = ReutersFactory(storage_connector)
        reuters_iterator, meta = reuters_factory.get_dataset_iterator(config={"split": "test"})
        sample, target, tag = reuters_iterator[0]
        print(sample)
        print(target)
