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


def _load_sample_paths(samples_identifier:str) -> List[str]:
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


def _load_metadata(targets_identifier:str) -> List[np.ndarray]:
    """
    function to load folder contents into lists in a sorted list
    @param targets_identifier: path to full_df.csv file
    @return: sortet list of lists conataining metadata
    """
    # Put rows as Tuples into lists and return them:
    counter = 0
    with open(targets_identifier, newline='') as targets:
        reader = csv.reader(targets)
        # From a row, it is needed the 17th and 18th position. Last two items
        targets_list = [row[17:19] for row in reader]

        # sort by file name ID -> target[18]
        targets_list = sorted(targets_list, key=lambda target: target[1])

        # delete last element, because it is the not needed header of csv
        # targets_list = targets_list[:3500]

        logger.debug(f'len(targets_list): {len(targets_list)}')
        logger.debug(f'targets_list[10][18]: {targets_list[10][0]}')
        logger.debug(f'len(targets_list[10]): {len(targets_list[1])}')
    return targets_list


def _get_most_common_res(samples:List[str]) -> Tuple[int, int]:
    samples_amount = len(samples)
    histo = list(range(samples_amount))

    for entry in range(samples_amount):
        with Image.open(samples[entry]) as img:
            width, height = img.size
            histo[entry] = (width, height)

    most_common = max(histo, key=histo.count)

    return most_common


def _get_clean_split_samples(resolution:tuple[int,int], split_samples) -> List[str]:
    cleaned_split_samples = []
    for entry in split_samples:
        with Image.open(entry) as img:
            if img.size == resolution:
                cleaned_split_samples.append(entry)

    return cleaned_split_samples


def _get_clean_split_targets(cleaned_split_samples:List[str], split_targets:List[List[str]]) -> list[tuple[Any, Any]]:
    cleaned_split_targets = []
    logger.debug(f"len(split_targets) in get clean targets list: {len(split_targets)}")

    for target_entry in split_targets:
        for sample_entry in cleaned_split_samples:
            file = Path(sample_entry).name
            if file == target_entry[1]:
                cleaned_split_targets.append((target_entry[1], target_entry[0]))

    return cleaned_split_targets


def _get_fin_clean_split_samples(cleaned_split_samples:List[str], cleaned_split_targets:List[List[str]]) -> List[str]:
    fin_clean_split_samples = []
    logger.debug(f"len(split_targets) in get clean targets list: {len(cleaned_split_targets)}")

    for sample_entry in cleaned_split_samples:
        for target_entry in cleaned_split_targets:
            file = Path(sample_entry).name
            if file == target_entry[0]:
                fin_clean_split_samples.append(sample_entry)

    return fin_clean_split_samples

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

    # challenge: find only samples, where the resolution is the same and the target is provided
    # 1) find resolution
    # 2) filter samples
    # 3) filter targets with filtered_samples
    # 4) filter filtered_samples with filtered_targets
    # 5) solution should be two csv files with only wanted and correct sorted information

    # samples resolution are not equal -> get the most common resolution
    logger.debug(f"_get_most_common_res(split_samples) starts")
    resolution = _get_most_common_res(split_samples)
    logger.debug(f"resolution {resolution}")

    # filter split_samples, so only wanted resolution is presented
    logger.debug(f"_get_clean_split_samples(resolution, split_samples) starts")
    cleaned_split_samples = _get_clean_split_samples(resolution, split_samples)
    logger.debug(f"len(cleaned_split_samples):{len(cleaned_split_samples)}")

    # filter split_targets, result will contain targets only for provided samples
    logger.debug(f"len(split_targets): {len(split_targets)}")
    logger.debug(f"_get_clean_split_targets(cleaned_split_samples, split_targets) starts")
    cleaned_split_targets = _get_clean_split_targets(cleaned_split_samples, split_targets)
    logger.debug(f"len(cleaned_split_targets):{len(cleaned_split_targets)}")

    # final filter on split samples, so it matches the filtered targets
    logger.debug(f"len(cleaned_split_samples):{len(cleaned_split_samples)}")
    logger.debug(f"_get_fin_clean_split_samples(cleaned_split_samples,cleaned_split_targets) starts")
    fin_clean_split_samples =_get_fin_clean_split_samples(cleaned_split_samples,cleaned_split_targets)
    logger.debug(f"len(fin_clean_split_samples):{len(fin_clean_split_samples)}")


    # sorting paths
    # sorting sample paths with natsort libriary
    logger.debug(f"natsorted(cleaned_split_samples) starts")
    sorted_fin_cleaned_split_samples = natsorted(fin_clean_split_samples)

    # target paths
    logger.debug(f"natsorted(split_targets) starts")
    sorted_cleaned_split_targets = natsorted(cleaned_split_targets)

    # create csv file with pandas
    # sample paths
    logger.debug(f"Create Pandas dataframe out of sample paths and save it as csv")
    df = pd.DataFrame(sorted_fin_cleaned_split_samples)
    df.to_csv('sorted_fin_cleaned_split_samples.csv', index=False, header=False)

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
