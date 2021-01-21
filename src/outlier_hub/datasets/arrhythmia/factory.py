#!/usr/bin/env python3

import os
from data_stack.io.retriever import RetrieverFactory
from data_stack.io.storage_connectors import StorageConnector, FileStorageConnector
from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.io.resources import StreamedTextResource
from outlier_hub.datasets.arrhythmia.preprocessor import ArrhythmiaPreprocessor
from outlier_hub.datasets.arrhythmia.iterator import ArrhythmiaIterator
from data_stack.io.resource_definition import ResourceDefinition
from data_stack.dataset.meta import MetaFactory
from typing import Dict, Any


class ArrhythmiaFactory(BaseDatasetFactory):

    def __init__(self, storage_connector: StorageConnector):
        self.raw_path = "arrhythmia/raw/"
        self.preprocessed_path = "arrhythmia/preprocessed/"
        self.resource_definitions = [
            ResourceDefinition(identifier=os.path.join(self.raw_path, "arrhythmia.data"),
                               source='https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data',
                               md5_sum="821c372c8c8886e29c9586fd1b81eb42"),
            ResourceDefinition(identifier=os.path.join(self.raw_path, "arrhythmia.names"),
                               source='https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.names',
                               md5_sum="3878879175cc2ce8f30a0ea55b8c4aa4")
        ]

        super().__init__(storage_connector)

    def check_exists(self) -> bool:
        # TODO come up with a better check!
        sample_identifier = self._get_resource_id(element="samples.pt")
        return self.storage_connector.has_resource(sample_identifier)

    def _get_resource_id(self, element: str) -> str:
        return os.path.join(self.preprocessed_path, element)

    def _retrieve_raw(self):
        retrieval_jobs = [ResourceDefinition(identifier=resource_definition.identifier,
                                             source=resource_definition.source,
                                             md5_sum=resource_definition.md5_sum)
                          for resource_definition in self.resource_definitions]
        retriever = RetrieverFactory.get_http_retriever(self.storage_connector)
        retriever.retrieve(retrieval_jobs)

    def _prepare(self):
        preprocessor = ArrhythmiaPreprocessor(self.storage_connector)
        sample_identifier = self._get_resource_id(element="samples.pt")
        target_identifier = self._get_resource_id(element="targets.pt")
        sample_resource, target_resource = preprocessor.preprocess(*[r.identifier for r in self.resource_definitions],
                                                                   sample_identifier=sample_identifier,
                                                                   target_identifier=target_identifier)
        self.storage_connector.set_resource(identifier=sample_resource.identifier, resource=sample_resource)
        self.storage_connector.set_resource(identifier=target_resource.identifier, resource=target_resource)

    def _get_iterator(self):
        sample_identifier = self._get_resource_id(element="samples.pt")
        target_identifier = self._get_resource_id(element="targets.pt")
        sample_resource = self.storage_connector.get_resource(identifier=sample_identifier)
        target_resource = self.storage_connector.get_resource(identifier=target_identifier)
        text_sample_resource = StreamedTextResource.from_streamed_resouce(sample_resource)
        text_target_resource = StreamedTextResource.from_streamed_resouce(target_resource)
        meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return ArrhythmiaIterator(text_sample_resource, text_target_resource), meta

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> DatasetIteratorIF:
        if not self.check_exists():
            self._retrieve_raw()
            self._prepare()
        return self._get_iterator()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        example_file_storage_path = os.path.join(root, "dataset_storage")
        storage_connector = FileStorageConnector(root_path=example_file_storage_path)

        arrhythmia_factory = ArrhythmiaFactory(storage_connector)
        arrhythmia_iterator, meta = arrhythmia_factory.get_dataset_iterator(config={"split": "full"})
        sample, target, _ = arrhythmia_iterator[0]
        print(sample)
        print(target)
