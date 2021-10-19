import os
import pathlib
from typing import Tuple, Dict, Any

from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.dataset.meta import IteratorMeta, MetaFactory
from data_stack.io.storage_connectors import StorageConnector, FileStorageConnector

from outlier_hub.datasets.ham10k.preprocessor import ODRPreprocessor
from outlier_hub.datasets.ham10k.iterator import ODRIterator


class ODRFactory(BaseDatasetFactory):
    pass
