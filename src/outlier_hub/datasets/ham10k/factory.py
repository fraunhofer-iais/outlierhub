#!/usr/bin/env python3

from typing import Tuple, Dict, Any
from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.io.storage_connectors import StorageConnector, FileStorageConnector
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.dataset.meta import IteratorMeta, MetaFactory
import os
import torchvision
import tempfile
from outlier_hub.datasets.ham10k.preprocessor import HAMPreprocessor
from outlier_hub.datasets.ham10k.iterator import HAMIterator

class Ham10kFactory(BaseDatasetFactory):

    def __init__(self, storage_connector: StorageConnector):
        self.splits = ['train']
        self.download_train_url = {'raw_images': {'url': 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip',
                                                  'file_name':'raw_images.zip'},
                                    'train_labels': {'url': 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip',
                                                  'file_name':'train_labels.zip'}
                                  }
        self.data_path = storage_connector.root_path
        self.dataset_name = 'isic2018.hdf5'

        super().__init__(storage_connector)

    def check_exists(self) -> bool:
        sample_identifier = self._get_resource_id(data_type = 'preprocessed', data_split='train/' + self.dataset_name)
        return self.storage_connector.has_resource(sample_identifier)

    def _get_resource_id(self, data_type: str, data_split: str) -> str:
        return os.path.join(self.data_path, data_type, data_split)

    def _retrieve_raw(self):
        for url_dict in self.download_train_url.values():

            train_path = self._get_resource_id(data_type='raw', data_split='train')

            torchvision.datasets.utils.download_url(url_dict['url'],
                                                    root=train_path,
                                                    filename = url_dict['file_name'])

            # train_path = os.path.join(train_path, url_dict['file_name'])                                        
            # archive = zipfile.ZipFile(train_path)
            # print(archive.namelist())


    def _prepare(self, split: str):
        preprocessor = HAMPreprocessor(self.storage_connector)
        raw_samples_identifier = self._get_resource_id(data_type = 'raw', data_split = split + '/raw_images.zip')
        raw_targets_identifier = self._get_resource_id(data_type = 'raw', data_split = split + '/train_labels.zip')

        preprocessed_dataset_identifier = self._get_resource_id(data_type='preprocessed', data_split='train/' + self.dataset_name)

        preprocessor.preprocess(raw_samples_identifier = raw_samples_identifier,
                                raw_targets_identifier = raw_targets_identifier,
                                prep_dataset_identifier = preprocessed_dataset_identifier)

    def _get_iterator(self, split: str):
        if not (self.check_exists()):
            print('Downloading and preparing the dataset.')
            if split not in self.splits:
                raise ResourceNotFoundError(f"Split {split} is not defined.")
            #self._retrieve_raw()
            self._prepare(split)
        dataset_identifier = self._get_resource_id(data_type = 'preprocessed', data_split = os.path.join(split, self.dataset_name))
        #dataset_resource = self.storage_connector.get_resource(identifier = dataset_identifier)
        meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return HAMIterator(dataset_identifier), meta
                            

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> Tuple[DatasetIteratorIF, IteratorMeta]:
        return self._get_iterator(**config)

if __name__ == "__main__":

    with tempfile.TemporaryDirectory() as root:
        example_file_storage_path = os.path.join(root, "dataset_storage")

        storage_connector = FileStorageConnector(root_path=example_file_storage_path)

        ham10k_factory = Ham10kFactory(storage_connector)

        # ham10k_factory._retrieve_raw()

        ham10k_iterator, _ = ham10k_factory.get_dataset_iterator(config={"split": "train"})
        
        print(len(ham10k_iterator))
        sample, target, tag = ham10k_iterator[200]
        print(sample.size())
        print(target)