import os
from typing import Tuple, Dict, Any

from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.dataset.meta import IteratorMeta, MetaFactory
from data_stack.io.storage_connectors import StorageConnector, FileStorageConnector

from preprocessor import HAMPreprocessor
from iterator import HAMIterator


class HAMFactory(BaseDatasetFactory):

    def __init__(self, storage_connector: StorageConnector):
        self.raw_path = os.getcwd()
        self.data_path = os.path.join(self.raw_path, "data")

        super().__init__(storage_connector)

    def check_exists(self) -> bool:
        # TODO come up with a better check!
        sample_identifier = self._get_resource_id(element="ham10k.hdf5")
        return self.storage_connector.has_resource(sample_identifier)

    def _get_resource_id(self, element: str) -> str:
        return os.path.join(self.data_path, element)

    def _prepare(self, split: str):
        # TODO like this it does not support train and test splits
        preprocessor = HAMPreprocessor(self.storage_connector)

        samples_identifier = self._get_resource_id(element=split + "/images")
        targets_identifier = self._get_resource_id(element=split + "/metadata/HAM10000_metadata.csv")
        dataset_identifier = self._get_resource_id(element="ham10k.hdf5")

        preprocessor.preprocess(dataset_identifier=dataset_identifier,
                                samples_identifier=samples_identifier,
                                targets_identifier=targets_identifier)

    def _get_iterator(self, split: str):
        dataset_identifier = self._get_resource_id(element="ham10k.hdf5")
        dataset_resource = self.storage_connector.get_resource(identifier=dataset_identifier)
        meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return HAMIterator(dataset_resource, split), meta

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> Tuple[DatasetIteratorIF, IteratorMeta]:
        if not (self.check_exists()):
            self._prepare(split="raw")
        return self._get_iterator(**config)


# Code for testing the dataset
if __name__ == "__main__":
    data_root = os.getcwd()

    data_storage_path = os.path.join(data_root, "data")

    storage_connector = FileStorageConnector(root_path=data_storage_path)

    ham_factory = HAMFactory(storage_connector)

    ham_iterator, _ = ham_factory.get_dataset_iterator(config={"split": "raw"})

    sample, target, tag = ham_iterator[1]

    print(type(ham_iterator[1]))
    print(type(sample))
