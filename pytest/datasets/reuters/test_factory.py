import pytest
import tempfile
import shutil
from data_stack.io.storage_connectors import StorageConnectorFactory
from outlier_hub.datasets.reuters.factory import ReutersFactory


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
        factory = ReutersFactory(storage_connector)
        iterator, meta = factory.get_dataset_iterator({"split": split_name})
        assert len(iterator) > 0
        assert iterator[0][0].shape[0] == 100
        assert isinstance(iterator[0][1], str) and isinstance(iterator[0][2], str)
