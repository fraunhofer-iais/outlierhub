from typing import Tuple, Dict, List
from data_stack.io.resources import ResourceFactory, StreamedResource, StreamedTextResource
from data_stack.io.storage_connectors import StorageConnector
import pandas as pd
import io


class KDDPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector

    def preprocess(self, preprocessed_train_identifier: str, preprocessed_test_identifier: str, features_and_target_names_identifier: str,
                   train_identifier: str, test_identifier: str) -> Tuple[StreamedResource]:

        # load dataframe columns
        with self.storage_connector.get_resource(features_and_target_names_identifier, ResourceFactory.SupportedStreamedResourceTypes.STREAMED_TEXT_RESOURCE) as features_and_target_names_resource:
            feature_to_type_mapping = KDDPreprocessor._get_features_and_types(features_and_target_names_resource)
            df_columns = list(feature_to_type_mapping.keys()) + ["target", "difficulty_level"]

        # load dataset splits
        with self.storage_connector.get_resource(train_identifier, ResourceFactory.SupportedStreamedResourceTypes.STREAMED_TEXT_RESOURCE) as train_resource:
            train_df = KDDPreprocessor._load_dataset_as_dataframe(train_resource, df_columns)
        with self.storage_connector.get_resource(test_identifier, ResourceFactory.SupportedStreamedResourceTypes.STREAMED_TEXT_RESOURCE) as test_resource:
            test_df = KDDPreprocessor._load_dataset_as_dataframe(test_resource, df_columns)

        # store dataset splits
        train_resource = KDDPreprocessor._data_frame_to_streamed_resource(preprocessed_train_identifier, train_df)
        test_resource = KDDPreprocessor._data_frame_to_streamed_resource(preprocessed_train_identifier, test_df)
        self.storage_connector.set_resource(preprocessed_train_identifier, train_resource)
        self.storage_connector.set_resource(preprocessed_test_identifier, test_resource)

    @staticmethod
    def _get_features_and_types(feature_types_resource: StreamedResource) -> Dict[str, str]:
        feature_types = {}
        features = feature_types_resource.readlines()[1:]  # the first line contains the targets, so we can skip it
        for feature in features:
            feature = feature.strip("\n")  # remove the newline character
            feature_name, feature_type = feature.split(":")  # split feature and type
            feature_type = feature_type.strip()
            feature_types[feature_name] = feature_type[0:len(feature_type) - 1]  # remove the extra dot at the end
        return feature_types

    @staticmethod
    def _load_dataset_as_dataframe(dataset_resource: StreamedTextResource, columns: List[str]) -> pd.DataFrame:
        dataset_df = pd.read_csv(dataset_resource, sep=",", header=None, names=columns)
        dataset_df = dataset_df.drop(axis=1, columns=["difficulty_level"])  # remove difficulty level
        return dataset_df

    def _data_frame_to_streamed_resource(identifier: str, df: pd.DataFrame) -> StreamedResource:
        string_buffer = io.StringIO()
        # TODO with to_pickle once https://github.com/pandas-dev/pandas/issues/35679 is fixed.
        df.to_csv(path_or_buf=string_buffer, index=False)
        string_buffer.seek(0)
        byte_buffer = io.BytesIO(string_buffer.read().encode('utf8'))
        resource = ResourceFactory.get_resource(identifier=identifier, file_like_object=byte_buffer)
        return resource
