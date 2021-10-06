import csv
import os
import glob
import tempfile
from typing import Tuple, List

import h5py
import matplotlib.image as mpimg
import numpy as np
from data_stack.io.resources import StreamedResource, ResourceFactory
from data_stack.io.storage_connectors import StorageConnector


class HAMPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector
        # TODO right now only one split is needed
        self.split_names = ["raw"]

    # def preprocess(self, img_resource: StreamedResource, metadata_resource: StreamedResource):
    def preprocess(self,
                   dataset_identifier: str,
                   samples_identifier: str,
                   targets_identifier: str) -> StreamedResource:
        with tempfile.TemporaryFile() as temp_file:
            with h5py.File(temp_file, 'w') as h5py_file:
                for split_name in self.split_names:
                    self._preprocess_split(h5py_file,
                                           split_name,
                                           temp_file,
                                           samples_identifier,
                                           targets_identifier)
                h5py_file.flush()
                temp_file.flush()
            streamed_resource = ResourceFactory.get_resource(dataset_identifier, temp_file)
            self.storage_connector.set_resource(dataset_identifier, streamed_resource)
            streamed_resource = self.storage_connector.get_resource(dataset_identifier)
        return streamed_resource

    def _preprocess_split(self,
                          h5py_file: h5py.File,
                          split_name: str,
                          temporary_file,
                          samples_identifier: str,
                          target_identifier: str):
        split_samples, split_targets = self._get_raw_dataset_split(split_name,samples_identifier,target_identifier)


        sample_location = os.path.join(split_name, "samples")
        target_location = os.path.join(split_name, "targets")

        # samples are images here, which can be intepreted as numpy ndarrays:
        # so a colored image has height*width pixels, each pixel contains three values representing RGB-Color
        height = 450
        width = 600
        rgb_channel = 3

        sample_dset = h5py_file.create_dataset(sample_location,
                                               shape=(len(split_samples), height, width, rgb_channel),
                                               dtype=int)

        # h5py cannot save np.ndarrays with strings by default, costum dtype must be created
        utf8_type = h5py.string_dtype('utf-8')

        # There are 8 meta information for each sample
        metadata_info_amount = 8

        target_dset = h5py_file.create_dataset(target_location,
                                               shape=(len(split_targets), metadata_info_amount,),
                                               dtype=utf8_type)

        for cnt, sample in enumerate(split_samples):
            sample_dset[cnt:cnt + 1, :, :] = mpimg.imread(sample)

        for cnt, target in enumerate(split_targets):
            target_dset[cnt] = target

    def _get_raw_dataset_split(self,
                               split_name: str,
                               samples_identifier: str,
                               target_identifier: str) -> Tuple[List[str], List[np.ndarray]]:
        """
        get tuple containing two lists, first inherits samples and second target information

        @param split_name: i.e. a string which defines 'train' or 'pytest' dataset
        @param samples_identifier: contains string to necessary images
        @param target_identifier: contains string to necessary metadata
        @return: returns a tuple which contains list od samples and list of targets
        """
        samples_identifier = samples_identifier
        targets_identifier = target_identifier

        def load_sample_paths(samples_identifier) -> List[str]:
            """
            function to load folder content into arrays and then it returns that same array
            @param samples_resource: path to samples, here i.e. images
            @return: sorted list of paths of raw samples
            """
            # Put filespaths  into lists and return them:
            raw_samples_paths = []
            for file in sorted(glob.glob(samples_identifier + '/*.jpg')):
                raw_samples_paths.append(file)

            print(f'Length Check of raw sample paths, should be 10015 and result is: \n {len(raw_samples_paths)}')

            return raw_samples_paths

        def load_metadata(targets_identifier) -> List[np.ndarray]:
            """
            function to load folder content into arrays and then it returns that same array
            @param targets_resource: path to metadata.csv file
            @return: sorted list with ISIC ID of metadata in tupels, each sample gets a Tupel with 8 entries
            """
            # Put rows as Tuples into lists and return them:
            with open(targets_identifier, newline='') as targets:
                reader = csv.reader(targets)
                targets_list = [np.array(row) for row in reader]
                # sort by ISIC ID -> target[1]
                targets_list = sorted(targets_list, key=lambda target: target[1])
                # delete last element, because it is the not needed header of csv
                targets_list = targets_list[:10015]

                print(f'Length Check of raw meta data, should be 10015 and result is: \n {len(targets_list)}')
                print(f'Length Check of single tuples, should be 8 and result is: \n {len(targets_list[0])}')

            return targets_list

        samples_resource= load_sample_paths(samples_identifier=samples_identifier)
        targets_resource= load_metadata(targets_identifier=targets_identifier)


        return samples_resource, targets_resource