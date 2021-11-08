import os
import pathlib
from typing import Tuple, Dict, Any
import PIL
from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.dataset.meta import IteratorMeta, MetaFactory
from data_stack.io.storage_connectors import StorageConnector, FileStorageConnector

from outlier_hub.datasets.odr.preprocessor import ODRPreprocessor
from outlier_hub.datasets.odr.iterator import ODRIterator
from data_stack.util.logger import logger


class ODRFactory(BaseDatasetFactory):

    def __init__(self, storage_connector: StorageConnector, ):
        # pathlib allows to manipulate the root folder, it takes it to the root directory of this project
        self.raw_path = pathlib.Path.cwd().parents[3]
        # completing the path to the manual added data source
        self.data_path = os.path.join(self.raw_path, "src/outlier_hub/datasets/odr/data")

        super().__init__(storage_connector)

    def check_exists(self) -> bool:
        # TODO come up with a better check!
        sample_identifier = "odr.hdf5"
        return self.storage_connector.has_resource(sample_identifier)

    def _get_resource_id(self, element: str) -> str:
        return os.path.join(self.data_path, element)

    def _prepare(self, split: str):
        preprocessor = ODRPreprocessor(self.storage_connector)

        samples_identifier = self._get_resource_id(element=split + "/images")
        targets_identifier = self._get_resource_id(element=split + "/metadata/data.xlsx")
        dataset_identifier = self._get_resource_id(element="odr.hdf5")

        logger.debug(f"preprocessor.preprocess(dataset/samples/targets - identifier) starts n"
                     f"{dataset_identifier, samples_identifier, targets_identifier}")

        preprocessor.preprocess(dataset_identifier=dataset_identifier,
                                samples_identifier=samples_identifier,
                                targets_identifier=targets_identifier)

    def _get_iterator(self, split: str):
        dataset_identifier = "odr.hdf5"
        dataset_resource = self.storage_connector.get_resource(identifier=dataset_identifier)
        meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return ODRIterator(dataset_resource, split), meta

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> Tuple[DatasetIteratorIF, IteratorMeta]:
        if not (self.check_exists()):
            self._prepare(split="raw")
        return self._get_iterator(**config)


# Code for testing the dataset
if __name__ == "__main__":
    logger.debug(f"starting event")

    # get root workind directory path
    root_path = pathlib.Path.cwd().parents[3]

    # complete path to manual added data
    data_path = os.path.join(root_path, "src/outlier_hub/datasets/odr/data")

    storage_connector = FileStorageConnector(root_path=data_path)

    odr_factory = ODRFactory(storage_connector)

    odr_iterator, _ = odr_factory.get_dataset_iterator(config={"split": "raw"})

    sample, target, tag = odr_iterator[2]
    print(target)
    sample[1].show()
