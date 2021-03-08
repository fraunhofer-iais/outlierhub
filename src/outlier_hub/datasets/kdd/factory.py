#!/usr/bin/env python3

import os
from data_stack.io.retriever import RetrieverFactory
from data_stack.io.storage_connectors import StorageConnector, FileStorageConnector
from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from outlier_hub.datasets.kdd.preprocessor import KDDPreprocessor
from outlier_hub.datasets.kdd.iterator import KDDIterator
from typing import List, Dict, Any
from data_stack.io.resource_definition import ResourceDefinition
from data_stack.dataset.meta import MetaFactory


class KDDFactory(BaseDatasetFactory):

    def __init__(self, storage_connector: StorageConnector, test_set_path: str, train_set_path: str, attack_type_mapping_path: str,
                 feature_and_target_names_path: str):
        super().__init__(storage_connector)
        self.raw_path = "kdd/raw/"  # this used for the resource definition to store the raw file via the storage connector
        self.preprocessed_path = "kdd/preprocessed/"

        self.attack_type_mapping_resouce_definition = ResourceDefinition(identifier=os.path.join(self.raw_path, "attack_type_mapping"),
                                                                         source=attack_type_mapping_path,
                                                                         md5_sum="4531577c083433973f38c0d6c50ee8eb")
        self.features_and_target_names_resouce_definition = ResourceDefinition(identifier=os.path.join(self.raw_path, "features_and_target_names"),
                                                                               source=feature_and_target_names_path,
                                                                               md5_sum="19e3ed2afd7b83e2268599816e973c63")
        self.test_set_resouce_definition = ResourceDefinition(identifier=os.path.join(self.raw_path, "test_set"),
                                                              source=test_set_path,
                                                              md5_sum="7c6d1b1af246690766394920d6b4c751")
        self.train_set_resouce_definition = ResourceDefinition(identifier=os.path.join(self.raw_path, "train_set"),
                                                               source=train_set_path,
                                                               md5_sum="f5592a95d1d1428348dfa6ca9652a800")

    def check_exists(self) -> bool:
        # TODO come up with a better check!
        sample_identifier = self._get_resource_id(element="kdd_dataset.hdf5")
        return self.storage_connector.has_resource(sample_identifier)

    def _get_resource_id(self, element: str) -> str:
        return os.path.join(self.preprocessed_path, element)

    def _retrieve_raw(self) -> List[str]:
        retrieval_jobs = [self.attack_type_mapping_resouce_definition, self.features_and_target_names_resouce_definition,
                          self.test_set_resouce_definition, self.train_set_resouce_definition]
        retriever = RetrieverFactory.get_file_retriever(self.storage_connector)
        return retriever.retrieve(retrieval_jobs)

    def _prepare(self):
        train_identifier = self._get_resource_id(element="train.pd")
        test_identifier = self._get_resource_id(element="test.pd")
        preprocessor = KDDPreprocessor(self.storage_connector)
        preprocessor.preprocess(preprocessed_train_identifier=train_identifier,
                                preprocessed_test_identifier=test_identifier,
                                features_and_target_names_identifier=self.features_and_target_names_resouce_definition.identifier,
                                train_identifier=self.train_set_resouce_definition.identifier,
                                test_identifier=self.test_set_resouce_definition.identifier)

    def _get_iterator(self, split: str):
        dataset_identifier = self._get_resource_id(element=f"{split}.pd")
        dataset_resource = self.storage_connector.get_resource(identifier=dataset_identifier)
        meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return KDDIterator(dataset_resource), meta

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> DatasetIteratorIF:
        """valid split names: train, test"""
        if not self.check_exists():
            self._retrieve_raw()
            self._prepare()
        return self._get_iterator(**config)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        example_file_storage_path = os.path.join(root, "dataset_storage")
        storage_connector = FileStorageConnector(root_path=example_file_storage_path)
        # TODO: The KDD dataset is not easily downloaded which is why we need to specifiy a file directory for the raw files.
        # import glob
        # raw_folder_path = ""
        # file_paths = sorted(glob.glob(os.path.join(raw_folder_path, "*")))
        # factory = KDDFactory(storage_connector, *file_paths)
        # iterator, meta = factory.get_dataset_iterator(config={"split": "train"})
        # sample, target, tag = iterator[0]
        # print(sample)
        # print(target)
