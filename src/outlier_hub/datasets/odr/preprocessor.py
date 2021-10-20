import os
import tempfile
import h5py
import glob
import csv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from collections import Counter
from typing import Tuple, List
from data_stack.io.resources import StreamedResource, ResourceFactory
from data_stack.io.storage_connectors import StorageConnector


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
        # samples resolution are not equal -> reduce all to 512x512

        samples_len = len(split_samples)
        # get a list with all resolutions as tupel with two integers
        histo = list(range(samples_len))

        for entry in range(samples_len):
            img = mpimg.imread(split_samples[entry])
            histo[entry] = img.shape[0:2]

        # transform histo into a list which contains only caclulated resolution values
        #ppi_set = [img[0] * img[1] for img in histo]
        # prepare plot
        bins = len(Counter(histo).keys())
        fig, ax = plt.subplots()
        ax.hist(histo, bins=bins)
        ax.ticklabel_format(style='plain', useOffset=False, axis='both')
        plt.show()

        # values, counts = np.unique(ppi_set, return_counts=True)
        # print(f'values: {values} \n counts: {counts}')


        # TODO: Images must be reduced

        height = 512
        width = 512
        rgb_channel = 3

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

            print(f'Length Check of raw sample paths, should be 7000 and result is: \n {len(raw_samples_paths)}')
            print(f'raw_samples_paths on point 10: {raw_samples_paths[10]}')
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

                print(f'Length Check of raw meta data, should be 3500 and result is: \n {len(targets_list)}')
                print(f'Checking on content point 10 and entry 10: \n {targets_list[10][18]}')
                print(f'Checking on content point 10 and entry 10: \n {targets_list[10][17]}')
                print(f'Checking on content point 10 and entry 10: \n {targets_list[10][16]}')
                print(f'Checking on length of an entry at point 10: \n {len(targets_list[10])}')
            return targets_list

        samples_resource = load_sample_paths(samples_identifier=samples_identifier)
        targets_resource = load_metadata(targets_identifier=targets_identifier)

        return samples_resource, targets_resource
