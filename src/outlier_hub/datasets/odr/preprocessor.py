import os
import tempfile
import h5py
import glob
import csv
import numpy as np
import pandas as pd

from natsort import natsorted, ns
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Any
from data_stack.io.resources import StreamedResource, ResourceFactory
from data_stack.io.storage_connectors import StorageConnector
from data_stack.util.logger import logger


def _load_sample_paths(samples_identifier) -> List[str]:
    """
    function to load folder content into arrays and then it returns that same array
    @param samples_identifier: path to samples, here i.e. images
    @return: sorted list of paths of raw samples
    """
    # Put file paths into lists and return them:
    raw_samples_paths = []
    for file in sorted(glob.glob(samples_identifier + '/*.jpg')):
        raw_samples_paths.append(file)

    logger.debug(f'Length Check of raw sample paths, should be 7000 and result is: \n {len(raw_samples_paths)}')
    logger.debug(f'raw_samples_paths on point 10: \n {raw_samples_paths[10]}')
    return raw_samples_paths


def _load_metadata(targets_identifier) -> List[np.ndarray]:
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

        logger.debug(f'len(targets_list): {len(targets_list)}')
        logger.debug(f'targets_list[10][18]: {targets_list[10][18]}')
        logger.debug(f'targets_list[10][17]: {targets_list[10][17]}')
        logger.debug(f'targets_list[10][16]: {targets_list[10][16]}')
        logger.debug(f'len(targets_list[10]): {len(targets_list[10])}')
    return targets_list


def _get_most_common_res(samples) -> Tuple[int, int]:
    # avoiding the use of list appending function, at first it will be created an empty list.
    samples_amount = len(samples)
    histo = list(range(samples_amount))

    for entry in range(samples_amount):
        with Image.open(samples[entry]) as img:
            width, height = img.size
            histo[entry] = (width, height)

    most_common = max(histo, key=histo.count)

    return most_common


def _get_clean_split_samples(resolution, split_samples) -> List[str]:
    cleaned_split_samples = []
    for entry in split_samples:
        with Image.open(entry) as img:
            if img.size == resolution:
                cleaned_split_samples.append(entry)

    return cleaned_split_samples


def _get_clean_split_targets(cleaned_split_samples, split_targets) -> list[tuple[Any, Any]]:
    cleaned_split_targets = []
    logger.debug(f"len(split_targets) in get clean targets list: {len(split_targets)}")
    for sample_entry in cleaned_split_samples:
        file = Path(sample_entry).name
        for target_entry in split_targets:
            if file == target_entry[18]:
                cleaned_split_targets.append((target_entry[18], target_entry[17]))

    return cleaned_split_targets


def _preprocess_split(h5py_file: h5py.File,
                      split_name: str,
                      samples_identifier: str,
                      targets_identifier: str):
    logger.debug(f"_preprocess_split(split_name,samples_identifier,targets_identifier) starts  "
                 f"{split_name, samples_identifier, targets_identifier}")

    logger.debug(f"load_sample_paths(samples_identifier) starts")
    split_samples = _load_sample_paths(samples_identifier)

    logger.debug(f"load_metadata(targets_identifier) starts")
    split_targets = _load_metadata(targets_identifier)

    # samples are images here, which can be intepreted as numpy 3D-arrays:
    # so a colored image has height*width pixels, each pixel contains three values representing RGB-Color
    # Each color of RGB is a so called channel [height, width, RGB-Value]

    # samples resolution are not equal -> get the most common resolution
    logger.debug(f"_get_most_common_res(split_samples) starts")
    resolution = _get_most_common_res(split_samples)
    logger.debug(f"resolution {resolution}")

    # manipulate split_samples, so only wanted resolution is presented
    logger.debug(f"_get_clean_split_samples(resolution, split_samples) starts")
    cleaned_split_samples = _get_clean_split_samples(resolution, split_samples)
    logger.debug(f"len(cleaned_split_samples):{len(cleaned_split_samples)}")

    # manipulate split_targets, so only wanted resolution is presented
    logger.debug(f"_get_clean_split_targets(cleaned_split_samples, split_targets) starts")
    logger.debug(f"len(split_targets): {len(split_targets)}")

    cleaned_split_targets = _get_clean_split_targets(cleaned_split_samples, split_targets)
    logger.debug(f"len(cleaned_split_targets):{len(cleaned_split_targets)}")

    # sorting paths
    # sorting sample paths with natsort libriary
    logger.debug(f"natsorted(cleaned_split_samples) starts")
    sorted_cleaned_split_samples = natsorted(cleaned_split_samples)

    # target paths
    logger.debug(f"natsorted(split_targets) starts")
    sorted_cleaned_split_targets = sorted(cleaned_split_targets)

    # create csv file with pandas
    # sample paths
    logger.debug(f"Create Pandas dataframe out of sample paths and save it as csv")
    df = pd.DataFrame(sorted_cleaned_split_samples)
    df.to_csv('sorted_cleaned_split_samples.csv', index=False, header=False)

    # target paths
    logger.debug(f"Create Pandas dataframe out of target paths and save it as csv")
    df = pd.DataFrame(sorted_cleaned_split_targets)
    df.to_csv('sorted_cleaned_split_targets.csv', index=False, header=False)


class ODRPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector
        self.split_names = ["raw"]

    def preprocess(self,
                   dataset_identifier: str,
                   samples_identifier: str,
                   targets_identifier: str) -> StreamedResource:
        logger.debug(f"preprocess(dataset/samples/targets - identifier) starts"
                     f"{dataset_identifier, samples_identifier, targets_identifier}")

        with tempfile.TemporaryFile() as temp_file:
            with h5py.File(temp_file, 'w') as h5py_file:
                for split_name in self.split_names:
                    _preprocess_split(h5py_file,
                                      split_name,
                                      samples_identifier,
                                      targets_identifier)
                h5py_file.flush()
                temp_file.flush()

            streamed_resource = ResourceFactory.get_resource(dataset_identifier, temp_file)

            self.storage_connector.set_resource(dataset_identifier, streamed_resource)

            streamed_resource = self.storage_connector.get_resource(dataset_identifier)

        return streamed_resource
