import csv
import os
import glob
import tempfile
import h5py
import matplotlib.image as mpimg
import numpy as np

from data_stack.util.logger import logger
from data_stack.io.resources import StreamedResource, ResourceFactory
from data_stack.io.storage_connectors import StorageConnector
from typing import Tuple, List
from pathlib import Path


def _get_raw_split_samples(samples_identifier: str) -> List[str]:
    """
    function to load folder content into arrays and then it returns that same array
    @param samples_identifier: path to samples, here i.e. images
    @return: sorted list of paths of raw samples
    """
    # Put filespaths  into lists and return them:
    raw_samples_paths = []
    for file in sorted(glob.glob(samples_identifier + '/*.jpg')):
        raw_samples_paths.append(file)
    logger.debug(f"Length Check - raw sample paths - 10015: {len(raw_samples_paths) == 10015}")

    return raw_samples_paths


def _get_raw_split_targets(target_identifier: str) -> List[np.ndarray]:
    """
    function to load folder content into arrays and then it returns that same array
    @param targets_identifier: path to metadata.csv file
    @return: sorted list with ISIC ID of metadata in tupels, each sample gets a Tupel with 8 entries
    """
    # Put rows as Tuples into lists and return them:
    with open(target_identifier, newline='') as targets:
        reader = csv.reader(targets)
        targets_list = [np.array(row) for row in reader]
        # sort by ISIC ID -> target[1]
        targets_list = sorted(targets_list, key=lambda target: target[1])
        # delete last element, because it is the not needed header of csv
        targets_list = targets_list[:10015]
        logger.debug(f"Length Check - raw meta data - 10015: {len(targets_list) == 10015}")
        logger.debug(f"Length Check - single tuples - 8: {len(targets_list[0]) == 8}")
        logger.debug(f"test print of a entry of target_list: \n {targets_list[0]}")

    return targets_list


def _preprocess_split(h5py_file: h5py.File,
                      split_name: str,
                      samples_identifier: str,
                      target_identifier: str):
    logger.debug(f"calling _preprocess_split(h5py_file, split_name, samples_identifier, target_identifier)")

    # split_samples, split_targets = _get_raw_dataset_split(samples_identifier, target_identifier)
    logger.debug(f" _get_raw_split_samples(samples_identifier: str) -> List[str]")
    split_samples = _get_raw_split_samples(samples_identifier)

    logger.debug(f" calling _get_raw_split_targets(target_identifier: str) -> List[np.ndarray]")
    split_targets = _get_raw_split_targets(target_identifier)

    sample_location = os.path.join(split_name, "samples")
    target_location = os.path.join(split_name, "targets")

    # create h5py groups, one for target and one for samples, every entry will be a dataset then
    logger.debug(f"Create h5py groups")
    sample_group = h5py_file.create_group('samples')
    target_group = h5py_file.create_group('targets')

    logger.debug(f" prepare and calling sample_dset = h5py_file.create_dataset")
    # samples are images here, which can be intepreted as numpy ndarrays:
    # so a colored image has height*width pixels, each pixel contains three values representing RGB-Color
    # height = 450
    # width = 600
    # rgb_channel = 3
    # sample_dset = h5py_file.create_dataset(sample_location,
    #                                        shape=(len(split_samples), height, width, rgb_channel),
    #                                        dtype=int)
    # logger.debug(f"enrich h5py_file with sample data")
    # for cnt, sample in enumerate(split_samples):
    #    sample_dset[cnt:cnt + 1, :, :] = mpimg.imread(sample)
    counter = 0
    for img_path in split_samples:

        img_name = Path(img_path).name

        # open image in binary, behind the path
        with open(img_path, 'rb') as img:
            img_binary = img.read()

        img_binary_np = np.asarray(img_binary)

        h5py_file = sample_group.create_dataset(str(counter), data=img_binary_np)
        counter = counter + 1

    logger.debug(f" prepare and calling target_dset = h5py_file.create_dataset")
    # h5py cannot save np.ndarrays with strings by default, costum dtype must be created
    utf8_type = h5py.string_dtype('utf-8')

    # There are 8 meta information for each sample
    # metadata_info_amount = 8

    # target_dset = h5py_file.create_dataset(target_location,
    #                                        shape=(len(split_targets), metadata_info_amount,),
    #                                       dtype=utf8_type)

    logger.debug(f"enrich h5py_file with sample data")
    counter = 0
    for target in split_targets:

        target = [str(item) for item in target]

        h5py_file = target_group.create_dataset(str(counter),
                                                data=target,
                                                dtype=utf8_type)
        counter = counter + 1
    # for cnt, target in enumerate(split_targets):
    #    target_dset[cnt] = target


class HAMPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector
        # TODO right now only one split is needed
        self.split_names = ["raw"]

    def preprocess(self,
                   dataset_identifier: str,
                   samples_identifier: str,
                   targets_identifier: str) -> StreamedResource:
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