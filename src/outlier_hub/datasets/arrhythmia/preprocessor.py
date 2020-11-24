from typing import Tuple
from data_stack.io.resources import ResourceFactory, StreamedResource
from data_stack.io.storage_connectors import StorageConnector
import pandas as pd
import io


class ArrhythmiaPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector

    def preprocess(self, raw_sample_identifier: str, raw_target_identifier: str, sample_identifier: str, target_identifier: str) -> Tuple[StreamedResource]:
        samples_df, targets_df = self._preprocess_raw(raw_sample_identifier, sample_identifier)
        sample_resource = self._data_frame_to_streamed_resource(sample_identifier, samples_df)
        target_resource = self._data_frame_to_streamed_resource(target_identifier, targets_df)
        return sample_resource, target_resource

    def _data_frame_to_streamed_resource(self, identifier: str, df: pd.DataFrame) -> StreamedResource:
        string_buffer = io.StringIO()
        # TODO with to_pickle once https://github.com/pandas-dev/pandas/issues/35679 is fixed.
        df.to_csv(path_or_buf=string_buffer, index=False)
        string_buffer.seek(0)
        byte_buffer = io.BytesIO(string_buffer.read().encode('utf8'))
        resource = ResourceFactory.get_resource(identifier=identifier, file_like_object=byte_buffer)
        return resource

    def _preprocess_raw(self, raw_identifier: str, prep_identifier: str) -> StreamedResource:
        with self.storage_connector.get_resource(raw_identifier) as raw_resource:
            data = pd.read_csv(raw_resource, header=None)
        columns = [f"f_{i}" for i in range(len(data.columns))][:-1] + ["target"]
        data.columns = columns
        data_clean = data.replace('?', 0)
        samples = data_clean.loc[:, data_clean.columns != 'target']
        targets = data_clean["target"]
        return samples, targets
