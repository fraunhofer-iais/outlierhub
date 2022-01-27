import os
import pathlib
from typing import Tuple, Dict, Any
import PIL
from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.dataset.meta import IteratorMeta, MetaFactory
from data_stack.io.storage_connectors import StorageConnector, FileStorageConnector

from preprocessor import ODRPreprocessor
from iterator import ODRIterator
from data_stack.util.logger import logger


class ODRFactory(BaseDatasetFactory):

    def __init__(self, storage_connector: StorageConnector, ):
        # pathlib allows to manipulate the root folder, it takes it to the root directory of this project
        logger.debug(f"self.raw_path = pathlib.Path.cwd() {pathlib.Path.cwd()}")
        self.raw_path = pathlib.Path.cwd()
        # completing the path to the manual added data source
        self.data_path = os.path.join(self.raw_path, "data")
        logger.debug(f"self.data_path  {self.data_path}")

        super().__init__(storage_connector)

    def check_exists(self) -> bool:
        # TODO come up with a better check!
        sample_identifier = "odr.hdf5"
        return self.storage_connector.has_resource(sample_identifier)

    def _get_resource_id(self, element: str) -> str:
        return os.path.join(self.data_path, element)

    def _prepare(self, split: str):
        preprocessor = ODRPreprocessor(self.storage_connector)
        print("test")
        samples_identifier = self._get_resource_id(element=split + "/images")
        targets_identifier = self._get_resource_id(element=split + "/metadata/data.xlsx")
        dataset_identifier = self._get_resource_id(element="odr.hdf5")

        logger.debug(f"preprocessor.preprocess(dataset/samples/targets - identifier) starts"
                     f"{dataset_identifier} \n , {samples_identifier},\n {targets_identifier}")

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
    root_path = pathlib.Path.cwd()

    # complete path to manual added data
    data_path = os.path.join(root_path, "data")
    print(f'data_path: {data_path}')

    storage_connector = FileStorageConnector(root_path=data_path)
    print(f'storage_connector: {storage_connector.root_path}')

    odr_factory = ODRFactory(storage_connector)

    odr_iterator, _ = odr_factory.get_dataset_iterator(config={"split": "raw"})

for i in range(5):
    sample, target, tag = odr_iterator[i]

    f = plt.figure(figsize=(15,15))

    f.add_subplot(1,2, 1)
    plt.title(target[3])
    plt.imshow(sample[0])

    f.add_subplot(1,2, 2)
    plt.title(target[4])
    plt.imshow(sample[1])
    plt.show(block=True)


    print(target)