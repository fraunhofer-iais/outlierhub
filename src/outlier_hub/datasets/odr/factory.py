import os
import pathlib
from typing import Tuple, Dict, Any

from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.dataset.meta import IteratorMeta, MetaFactory
from data_stack.io.storage_connectors import StorageConnector, FileStorageConnector

from outlier_hub.datasets.odr.preprocessor import ODRPreprocessor
from outlier_hub.datasets.odr.iterator import ODRIterator


class ODRFactory(BaseDatasetFactory):
    def __init__(self, storage_connector: StorageConnector,):
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
        targets_identifier = self._get_resource_id(element=split + "/metadata/HAM10000_metadata.csv")
        dataset_identifier = self._get_resource_id(element="ham10k.hdf5")

        preprocessor.preprocess(dataset_identifier=dataset_identifier,
                                samples_identifier=samples_identifier,
                                targets_identifier=targets_identifier)