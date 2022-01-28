import os
import sys
import tempfile
import h5py
import glob
import csv
import numpy as np
import pandas as pd
import io

from natsort import natsorted, ns
from PIL import Image, ImageFile
from pathlib import Path
from typing import Tuple, List, Any
from data_stack.io.resources import StreamedResource, ResourceFactory
from data_stack.io.storage_connectors import StorageConnector
from data_stack.util.logger import logger


def _load_sample_paths(samples_identifier: str) -> List[str]:
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


def _load_metadata(targets_identifier: str) -> List[np.ndarray]:
    """
    loads xlsx file, and creates a list - each item is a list with 15 items
    @param: targets_identifier: path to data.xlsx file
    @return: list of lists containing 15 items
    """
    # Use pandas to read and manipulate metadata:
    data_xls = pd.read_excel(targets_identifier, 'Sheet1', index_col=None, engine='openpyxl')
    # merge diagnostics
    data_xls["diagnostics"] = data_xls["Left-Diagnostic Keywords"] + ', ' + data_xls["Right-Diagnostic Keywords"]
    data_xls.drop("Left-Diagnostic Keywords", inplace=True, axis=1)
    data_xls.drop("Right-Diagnostic Keywords", inplace=True, axis=1)
    # rearrange columns
    cols = data_xls.columns.tolist()
    columns = ['ID', 'Patient Age', 'Patient Sex', 'Left-Fundus', 'Right-Fundus', 'diagnostics',
               'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    data_xls = data_xls[columns]
    # get a list of metadata
    data_values = data_xls.values.tolist()

    return data_values


def _get_most_common_res(samples: List[str]) -> Tuple[int, int]:
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


def _create_temp_list(split_targets, split_samples) -> List[Any]:
    # comprehension list to initiate temp_list with meta data
    temp_list = [[entry, [None, None]] for entry in split_targets]

    for item in temp_list:
        for sample_item in split_samples:
            file = Path(sample_item).name
            if file == item[0][3]:
                item[1][0] = sample_item
            elif file == item[0][4]:
                item[1][1] = sample_item
    return temp_list


def _clean_temp_list(temp_list):
    clean_temp_list = []
    for item in temp_list:
        if not (item[1] == [None, None]):
            clean_temp_list.append(item)

    return clean_temp_list


def _preprocess_split(h5py_file: h5py.File,
                      split_name: str,
                      samples_identifier: str,
                      targets_identifier: str):
    logger.debug(f"calling load_sample_paths(samples_identifier)")
    split_samples = _load_sample_paths(samples_identifier)

    logger.debug(f"calling load_metadata(targets_identifier)")
    split_targets = _load_metadata(targets_identifier)

    # sample information are paths to images
    # samples are at the end images here, which can be interpreted as numpy 3D-arrays:
    # so a colored image has height*width pixels, each pixel contains three values representing RGB-Color
    # Each color of RGB is a so called channel [height, width, RGB-Value]

    # necessary: find only samples, where the resolution is the same and the target is provided
    # 1) find resolution and filter samples with estimated resolution
    # 2) create new temp List for indexing targets to associated samples [meta,(sample1,sample2)]
    # 3) clean temp List from empty samples
    # 4) divide temp list into target and samples
    # 5) create csv files for manual verification of data
    # 6) prepare hdf datasets
    # 7) enrich datasets with data

    # 1) find resolution
    # samples resolution are not equal -> get the most common resolution
    logger.debug(f"calling  _get_most_common_res(split_samples)")
    resolution = _get_most_common_res(split_samples)
    logger.debug(f"resolution {resolution}")

    # filter split_samples, so only wanted resolution is provided
    logger.debug(f"_get_clean_split_samples(resolution, split_samples) starts")
    cleaned_split_samples = _get_clean_split_samples(resolution, split_samples)
    logger.debug(f"len(cleaned_split_samples):{len(cleaned_split_samples)}")

    # 2) create temp list : [meta,(sample1,sample2)]
    # 1. item contains meta info, 2. info is a tuple inheriting associated sample paths
    logger.debug(f"length & type of split_targets: {len(split_targets)}, {type(split_targets)} ")
    logger.debug(f"function calling: _create_temp_list(split_targets)")
    temp_list = _create_temp_list(split_targets, cleaned_split_samples)

    # 3) clean temp List from empty samples
    print(f"clean temp_list from empty targets without samples")
    print(f"length of temp_list: {len(temp_list)}")
    logger.debug(f"calling _clean_temp_list")
    clean_temp_list = _clean_temp_list(temp_list)
    print(f"length of clean_temp_list: {len(clean_temp_list)}")

    # 4) divide the list into target and sample list
    targets_list = [entry[0] for entry in clean_temp_list]
    samples_list = [entry[1] for entry in clean_temp_list]

    logger.debug(f"Create Pandas dataframe of clean_temp_list list and save it as csv")
    df = pd.DataFrame(temp_list)
    df.to_csv('clean_temp_list.csv', index=False, header=False)

    # 5) create csv file with pandas - this is for manual verification
    logger.debug(f"Create Pandas dataframe of target and samples list and save it as csv")
    df = pd.DataFrame(targets_list)
    df.to_csv('targets_list.csv', index=False, header=False)
    df = pd.DataFrame(samples_list)
    df.to_csv('samples_list.csv', index=False, header=False)

    # create h5py groups, one for target and one for samples, every entry will be a dataset then
    logger.debug(f"Create h5py groups")
    sample_group = h5py_file.create_group('samples')
    target_group = h5py_file.create_group('targets')

    # saving the images in samples group
    counter = 0

    for entry in samples_list:
        eyepair_sample_group = sample_group.create_group(str(counter))
        counter = counter + 1

        img_path_1 = entry[0]
        img_name_1 = Path(entry[0]).name

        img_path_2 = entry[1]
        img_name_2 = Path(entry[1]).name

        # open image, behind the path
        with open(img_path_1, 'rb') as img:
            binary_data_1 = img.read()

        with open(img_path_2, 'rb') as img:
            binary_data_2 = img.read()

        binary_data_np_1 = np.asarray(binary_data_1)
        binary_data_np_2 = np.asarray(binary_data_2)

        # save it in the subgroup. each eyepair_sample_group contains images from one patient.
        h5py_file = eyepair_sample_group.create_dataset(img_name_1, data=binary_data_np_1)
        h5py_file = eyepair_sample_group.create_dataset(img_name_2, data=binary_data_np_2)

    # saving the targets in targets group
    # h5py cannot save np.ndarrays with strings by default, costum dtype must be created
    utf8_type = h5py.string_dtype('utf-8')

    # metadata_info_amount = 14

    counter = 0
    for entry in targets_list:

        entry = [str(item) for item in entry]

        h5py_file = target_group.create_dataset(str(counter),
                                                data=entry,
                                                dtype=utf8_type)
        counter = counter + 1

    '''
    # paths for h5py
    sample_location = os.path.join(split_name, "samples")
    target_location = os.path.join(split_name, "targets")

    # 6) prepare hdf datasets
    # with open(samples_list[0][0], 'rb') as image:
    #    type_reference = image.read()
    #    print(f'type(image.read(): {type(image.read())}')
    #    data_shape = np.asarray(image.read())

    # sample_dset = h5py_file.create_dataset(sample_location,
    #                                       shape=(len(clean_temp_list), [data_shape, data_shape]),
    #                                       dtype=np.void(type_reference))

    # 7) enrich datasets with data
    sample_dset = h5py_file.create_dataset(sample_location,
                                           data=samples_list)
    
    for cnt, sample in enumerate(samples_list):
        tmp_sample_list = []

        with open(sample[0], 'rb') as image_sample:
            sample_left_bytes = image_sample.read()
        tmp_sample_list.append(sample_left_bytes)

        with open(sample[1], 'rb') as image_sample:
            sample_right_bytes = image_sample.read()
        tmp_sample_list.append(sample_right_bytes)

        sample_pair_np = np.asarray(tmp_sample_list)
        sample_dset[cnt, 1] = sample_pair_np

        #logger.debug(f" testimage")
        #sample_np = sample_dset[cnt, 1]
        #sample_bytes = sample_np.tobytes()
        #sample_bytes = io.BytesIO(sample_bytes)
        #sample = Image.open(sample_bytes)
        #sample.show()
        
            for cnt, target in enumerate(targets_list):
        target_dset[cnt] = np.array(target)
    '''


class ODRPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector
        self.split_names = ["raw"]

    def preprocess(self,
                   dataset_identifier: str,
                   samples_identifier: str,
                   targets_identifier: str) -> StreamedResource:
        logger.debug(f"preprocess(dataset/samples/targets - identifier) starts"
                     f"{dataset_identifier} \n , {samples_identifier},\n {targets_identifier}")

        with tempfile.TemporaryFile() as temp_file:
            with h5py.File(temp_file, 'w') as h5py_file:
                for split_name in self.split_names:
                    _preprocess_split(h5py_file,
                                      split_name,
                                      samples_identifier,
                                      targets_identifier)
                h5py_file.flush()
                temp_file.flush()

            logger.debug(f"ResourceFactory.get_resource(dataset_identifier, temp_file) starts"
                         f"{dataset_identifier, samples_identifier, targets_identifier}")

            streamed_resource = ResourceFactory.get_resource(dataset_identifier, temp_file)

            self.storage_connector.set_resource(dataset_identifier, streamed_resource)

            streamed_resource = self.storage_connector.get_resource(dataset_identifier)

        return streamed_resource
