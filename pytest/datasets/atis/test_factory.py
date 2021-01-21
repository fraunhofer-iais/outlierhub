import pytest
import tempfile
import shutil
from data_stack.io.storage_connectors import StorageConnectorFactory
from outlier_hub.datasets.atis.factory import AtisFactory
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

    @pytest.mark.parametrize("split_name", ["train", "val", "test"])
    def test_get_dataset_iterator(self, storage_connector, split_name):
        # TODO: The ATIS dataset is not easily downloaded which is why we need to specifiy a file directory for the raw files. 
        # raw_folder_path = 
        # file_paths = sorted(glob.glob(os.path.join(raw_folder_path, "*")))
        # factory = AtisFactory(storage_connector, *file_paths)
        # iterator, meta = factory.get_dataset_iterator({"split": split_name})
        # assert len(iterator) == 4274 if split_name == "train" else True
        # assert len(iterator) == 572 if split_name == "val" else True
        # assert len(iterator) == 586 if split_name == "test" else True
        # assert iterator[0][0].shape[0] == 100
        # assert isinstance(iterator[0][1], str) and isinstance(iterator[0][2], str)
        pass
