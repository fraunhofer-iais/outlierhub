
import os
import glob
import tempfile
import matplotlib.image as mpimg
from typing import Optional, Tuple, List
import numpy as np
from zipfile import ZipFile
from PIL import Image
from torchvision import transforms
import h5py
import pandas as pd
from data_stack.io.resources import StreamedResource, ResourceFactory
from data_stack.io.storage_connectors import StorageConnector
from pathlib import Path

class HAMPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector

    def unzip_samples(self, sample_file: str):

        new_filepath = sample_file.split('.')[0]

        zip_file = ZipFile(sample_file, 'r')

        zip_infos = zip_file.infolist()

        for zipinfo in zip_infos:
            archive_filename = zipinfo.filename

            zipinfo.filename = os.path.basename( zipinfo.filename)

            if not zipinfo.filename == '' :
                zip_file.extract(member = archive_filename, path = new_filepath)

        zip_file.close()

        os.remove(sample_file)

        return new_filepath

    def preprocess(self,
                   raw_samples_identifier: str,
                   raw_targets_identifier: str,
                   prep_dataset_identifier: str):

        self.split_names = ["raw"]

        # following line is needed, if data is not already retrieved
        raw_samples_dir = self.unzip_samples(raw_samples_identifier)
        raw_targets_dir = self.unzip_samples(raw_targets_identifier)

        # following line is needed, when data is already provided
        #raw_samples_identifier = raw_samples_identifier.split('.')[0]
        #raw_samples_identifier = raw_samples_identifier.split('.')[0]

        if not os.path.exists(prep_dataset_identifier):
            os.makedirs(os.path.dirname(prep_dataset_identifier), exist_ok=True)


        with tempfile.TemporaryFile() as temp_file:
            with h5py.File(temp_file, 'w') as h5py_file:
                self.prepare_dataset(h5py_file,
                                    raw_samples_dir,
                                    raw_targets_dir)
                h5py_file.flush()
                temp_file.flush()
            streamed_resource = ResourceFactory.get_resource(prep_dataset_identifier, temp_file)
            self.storage_connector.set_resource(prep_dataset_identifier, streamed_resource)
            streamed_resource = self.storage_connector.get_resource(prep_dataset_identifier)

        return streamed_resource

    def prepare_dataset(self,
                        h5py_file : h5py.File,
                        raw_samples_identifier,
                        raw_targets_identifier):

        df = pd.read_csv('src/outlier_hub/datasets/ham10k/data/raw/train/train_labels/ISIC2018_Task3_Training_GroundTruth.csv', header=0)
        
        print(df.head())

        df = df.sort_values(by = 'image')
        labels_arr = df.iloc[:, 1:len(df.columns)].values

        target_dset = h5py_file.create_dataset('targets', data = labels_arr)

        sample_group = h5py_file.create_group('samples')

        print(f'raw_samples_identifier): {raw_samples_identifier}')
        split_samples = sorted(glob.glob(raw_samples_identifier + '/*.jpg'))
        print(f'len(split_samples): {len(split_samples)}')

        counter = 0
        for img_path in split_samples:

            # open image in binary, behind the path
            with open(img_path, 'rb') as img:
                img_binary = img.read()

            img_binary_np = np.asarray(img_binary)

            h5py_file = sample_group.create_dataset(str(counter), data=img_binary_np)
            counter = counter + 1