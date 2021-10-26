import os
import tempfile
import timeit

import h5py
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

from collections import Counter
from typing import Tuple, List
from data_stack.io.resources import StreamedResource, ResourceFactory
from data_stack.io.storage_connectors import StorageConnector
from data_stack.util.logger import logger


def _get_most_common_res(samples):
    # avoiding the use of list appending function, at first it will be created an empty list.
    samples_amount = len(samples)
    histo = list(range(samples_amount))

    for entry in range(samples_amount):
        img = mpimg.imread(samples[entry])
        histo[entry] = img.shape[0:2]

    most_common = max(histo, key=histo.count)

    return most_common


def _get_clean_split_samples(resolution, split_samples):
    cleaned_split_samples = []
    for entry in split_samples:
        img = mpimg.imread(entry)
        if img.shape[0:2] == resolution:
            cleaned_split_samples.append(entry)

    return cleaned_split_samples


class ODRPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector
        self.split_names = ["raw"]

    def preprocess(self,
                   dataset_identifier: str,
                   samples_identifier: str,
                   targets_identifier: str) -> StreamedResource:
        with tempfile.TemporaryFile() as temp_file:
            with h5py.File(temp_file, 'w') as h5py_file:
                for split_name in self.split_names:
                    self._preprocess_split(h5py_file,
                                           split_name,
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
                          samples_identifier: str,
                          target_identifier: str):

        split_samples, split_targets = self._get_raw_dataset_split(samples_identifier, target_identifier)

        sample_location = os.path.join(split_name, "samples")
        target_location = os.path.join(split_name, "targets")

        # samples are images here, which can be intepreted as numpy ndarrays:
        # so a colored image has height*width pixels, each pixel contains three values representing RGB-Color
        # samples resolution are not equal -> get the most common resolution

        resolution = _get_most_common_res(split_samples)
        logger.debug(f"resolution {resolution}")
        # manipualte split_samples, so only wanted resolution is presented

        cleaned_split_samples = _get_clean_split_samples(resolution, split_samples)
        logger.debug(f"len(cleaned_split_samples) {len(cleaned_split_samples)}")
        print(cleaned_split_samples)
        df = pd.DataFrame(cleaned_split_samples)
        result = df[0].value_counts()
        result.to_csv('test.csv')

    def _get_raw_dataset_split(self,
                               samples_identifier: str,
                               target_identifier: str) -> Tuple[List[str], List[np.ndarray]]:
        """
        get tuple containing two lists, first inherits samples and second target information
        @param samples_identifier: contains string to necessary images
        @param target_identifier: contains string to necessary metadata
        @return: returns a tuple which contains list od samples and list of targets
        """
        samples_identifier = samples_identifier
        targets_identifier = target_identifier

        def load_sample_paths(samples_identifier) -> List[str]:
            """
            function to load folder content into arrays and then it returns that same array
            @param samples_identifier: path to samples, here i.e. images
            @return: sorted list of paths of raw samples
            """
            # Put filespaths  into lists and return them:
            raw_samples_paths = []
            for file in sorted(glob.glob(samples_identifier + '/*.jpg')):
                raw_samples_paths.append(file)

            logger.debug(f'Length Check of raw sample paths, should be 7000 and result is: \n {len(raw_samples_paths)}')
            logger.debug(f'raw_samples_paths on point 10: \n {raw_samples_paths[10]}')
            return raw_samples_paths

        def load_metadata(targets_identifier) -> List[np.ndarray]:
            """
            function to load folder content into arrays and then it returns that same array
            @param targets_identifier: path to metadata.csv file
            @return: sorted list with file name of metadata in tupels, it should only contain
            """
            # Put rows as Tuples into lists and return them:
            with open(targets_identifier, newline='') as targets:
                reader = csv.reader(targets)
                targets_list = [np.array(row) for row in reader]
                # sort by file name ID -> target[18]
                targets_list = sorted(targets_list, key=lambda target: target[18])
                # delete last element, because it is the not needed header of csv
                targets_list = targets_list[:3500]

                logger.debug(f'Length Check of raw meta data, should be 3500 and result is: \n {len(targets_list)}')
                logger.debug(f'Checking on content point 10 and entry 10: \n {targets_list[10][18]}')
                logger.debug(f'Checking on content point 10 and entry 10: \n {targets_list[10][17]}')
                logger.debug(f'Checking on content point 10 and entry 10: \n {targets_list[10][16]}')
                logger.debug(f'Checking on length of an entry at point 10: \n {len(targets_list[10])}')
            return targets_list

        samples_resource = load_sample_paths(samples_identifier=samples_identifier)
        targets_resource = load_metadata(targets_identifier=targets_identifier)

        return samples_resource, targets_resource
