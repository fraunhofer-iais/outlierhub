import os
import logging
import pathlib
import random

from data_stack.io.storage_connectors import StorageConnectorFactory
from outlier_hub.datasets.odr.factory import ODRFactory
from outlier_hub.datasets.odr.iterator import ODRIterator
import pytest


class TestFactory:

    @pytest.fixture
    def tmp_folder_path(self) -> str:
        # get root working directory path
        root_path = pathlib.Path.cwd()

        # complete path to manual added data
        data_path = os.path.join(root_path, "src/outlier_hub/datasets/odr/data")
        return data_path

    @pytest.fixture
    def storage_connector(self, tmp_folder_path):
        return StorageConnectorFactory.get_file_storage_connector(tmp_folder_path)

    @pytest.mark.parametrize("split_name", ["raw"])
    def test_get_dataset_iterator(self, storage_connector, split_name):
        factory = ODRFactory(storage_connector)

        iterator, _ = factory.get_dataset_iterator(config={"split": "raw"})
        sample, target, _ = iterator[random.randint(0, 1014)]

        # checking the type and size of whole iterator
        assert isinstance(iterator, ODRIterator)
        assert len(iterator) > 0
        assert len(iterator) == 7000

