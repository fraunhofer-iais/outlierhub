#!/usr/bin/env python3

import os
from data_stack.io.storage_connectors import StorageConnector, FileStorageConnector
from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from outlier_hub.datasets.newsgroups.preprocessor import NewsGroupsPreprocessor
from outlier_hub.datasets.newsgroups.iterator import NewsGroupsIterator
from data_stack.dataset.meta import MetaFactory
from typing import Dict, Any
import tempfile


class NewsGroupsFactory(BaseDatasetFactory):
    """
    Newsgroups dataset containing messages from 20 different classes.
    For reference, see: http://qwone.com/~jason/20Newsgroups/
    """

    def __init__(self, storage_connector: StorageConnector):
        self.preprocessed_path = "reuters/preprocessed/"

        super().__init__(storage_connector)

    def check_exists(self) -> bool:
        # TODO come up with a better check!
        sample_identifier = self._get_resource_id(element="news_groups.hdf5")
        return self.storage_connector.has_resource(sample_identifier)

    def _get_resource_id(self, element: str) -> str:
        return os.path.join(self.preprocessed_path, element)

    def _prepare(self):
        dataset_identifier = self._get_resource_id(element="news_groups.hdf5")
        preprocessor = NewsGroupsPreprocessor(self.storage_connector)
        preprocessor.preprocess(dataset_identifier=dataset_identifier)

    def _get_iterator(self):
        dataset_identifier = self._get_resource_id(element="news_groups.hdf5")
        dataset_resource = self.storage_connector.get_resource(identifier=dataset_identifier)
        meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return NewsGroupsIterator(dataset_resource), meta

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> DatasetIteratorIF:
        if not self.check_exists():
            self._prepare()
        return self._get_iterator()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as root:
        example_file_storage_path = os.path.join(root, "dataset_storage")
        storage_connector = FileStorageConnector(root_path=example_file_storage_path)

        reuters_factory = NewsGroupsFactory(storage_connector)
        iterator, meta = reuters_factory.get_dataset_iterator()
        sample, target, tag = iterator[0]
        print(sample)
        print(target)
        from collections import Counter
        targets = [t for _, t, _ in iterator]
        counts = dict(Counter(targets))
        print(counts)
