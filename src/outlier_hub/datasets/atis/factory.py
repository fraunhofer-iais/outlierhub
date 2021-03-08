#!/usr/bin/env python3

import os
from data_stack.io.storage_connectors import StorageConnector, FileStorageConnector
from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from outlier_hub.datasets.atis.preprocessor import AtisPreprocessor
from outlier_hub.datasets.atis.iterator import AtisIterator
from data_stack.io.resource_definition import ResourceDefinition
from data_stack.io.retriever import RetrieverFactory
from data_stack.dataset.meta import MetaFactory
from typing import Dict, Any


class AtisFactory(BaseDatasetFactory):

    def __init__(self, storage_connector: StorageConnector, val_set_path: str, test_set_path: str, train_set_path: str):
        self.raw_path = "atis/raw/"
        self.preprocessed_path = "atis/preprocessed/"

        self.train_set_resouce_definition = ResourceDefinition(identifier=os.path.join(self.raw_path, "train_set"),
                                                               source=train_set_path,
                                                               md5_sum="a83554393ec93b94b6241aded74c9e2b")
        self.val_set_resouce_definition = ResourceDefinition(identifier=os.path.join(self.raw_path, "val_set"),
                                                             source=val_set_path,
                                                             md5_sum="372bf2a36821ed5c6d9db39ad19f5e1f")
        self.test_set_resouce_definition = ResourceDefinition(identifier=os.path.join(self.raw_path, "test_set"),
                                                              source=test_set_path,
                                                              md5_sum="60d01cfbef5f7f84fe37d7782c3051e5")
        super().__init__(storage_connector)

    def check_exists(self) -> bool:
        # TODO come up with a better check!
        sample_identifier = self._get_resource_id(element="atis_dataset.hdf5")
        return self.storage_connector.has_resource(sample_identifier)

    def _get_resource_id(self, element: str) -> str:
        return os.path.join(self.preprocessed_path, element)

    def _retrieve_raw(self):
        retrieval_jobs = [self.train_set_resouce_definition, self.val_set_resouce_definition, self.test_set_resouce_definition]
        retriever = RetrieverFactory.get_file_retriever(self.storage_connector)
        return retriever.retrieve(retrieval_jobs)

    def _prepare(self):
        dataset_identifier = self._get_resource_id(element="atis_dataset.hdf5")
        preprocessor = AtisPreprocessor(self.storage_connector)
        preprocessor.preprocess(preprocessed_dataset_identifier=dataset_identifier,
                                raw_train_identifier=self.train_set_resouce_definition.identifier,
                                raw_val_identifier=self.val_set_resouce_definition.identifier,
                                raw_test_identifier=self.test_set_resouce_definition.identifier)

    def _get_iterator(self, split: str):
        """Supported splits: train, val, test
        """
        dataset_identifier = self._get_resource_id(element="atis_dataset.hdf5")
        dataset_resource = self.storage_connector.get_resource(identifier=dataset_identifier)
        meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return AtisIterator(dataset_resource, split), meta

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
        # TODO: The ATIS dataset is not easily downloaded which is why we need to specifiy a file directory for the raw files.
        # import glob
        # raw_folder_path = ""
        # file_paths = sorted(glob.glob(os.path.join(raw_folder_path, "*")))
        # factory = AtisFactory(storage_connector, *file_paths)
        # iterator, meta = factory.get_dataset_iterator(config={"split": "train"})
        # sample, target, tag = iterator[0]
        # print(sample)
        # print(target)
