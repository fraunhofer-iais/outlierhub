import pytest
import tempfile
import shutil
from data_stack.io.storage_connectors import StorageConnectorFactory
from outlier_hub.datasets.kdd.factory import KDDFactory
import glob
import os


class TestFactory:

    @pytest.fixture
    def tmp_folder_path(self) -> str:
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)

    @pytest.fixture
    def storage_connector(self, tmp_folder_path):
        return StorageConnectorFactory.get_file_storage_connector(tmp_folder_path)

    @pytest.mark.parametrize("split_name", ["train", "test"])
    def test_get_dataset_iterator(self, storage_connector, split_name):
        # TODO: The KDD dataset is not easily downloaded which is why we need to specifiy a file directory for the raw files.
        # raw_folder_path = ""
        # file_paths = sorted(glob.glob(os.path.join(raw_folder_path, "*")))
        # factory = KDDFactory(storage_connector, *file_paths)
        # iterator, meta = factory.get_dataset_iterator(split_name)
        # assert len(iterator) > 0
        # assert iterator[0][0].shape[0] == 41
        # assert isinstance(iterator[0][1], str) and isinstance(iterator[0][2], str)
