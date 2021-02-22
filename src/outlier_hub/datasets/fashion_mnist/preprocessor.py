import torch
from data_stack.io.resources import ResourceFactory, StreamedResource
import io
from data_stack.io.storage_connectors import StorageConnector
from torchvision import transforms


class FashionMNISTPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector

    def preprocess(self, raw_identifier: str, preprocessed_samples_identifier: str, preprocessed_targets_identifier: str):
        with self.storage_connector.get_resource(raw_identifier) as raw_resource:
            samples, targets = torch.load(raw_resource)
            with self._preprocess_sample_resource(samples, preprocessed_samples_identifier) as sample_resource:
                self.storage_connector.set_resource(identifier=sample_resource.identifier, resource=sample_resource)
            with self._preprocess_target_resource(targets, preprocessed_targets_identifier) as target_resource:
                self.storage_connector.set_resource(identifier=target_resource.identifier, resource=target_resource)

    def _torch_tensor_to_streamed_resource(self, identifier: str, tensor: torch.Tensor) -> StreamedResource:
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        resource = ResourceFactory.get_resource(identifier=identifier, file_like_object=buffer)
        return resource

    def _preprocess_sample_resource(self, samples: torch.Tensor, preprocessed_identifier: str) -> StreamedResource:
        samples = samples.float()
        # we don't calculate this on the fly, since the normalization values are calculated on train and then applied to train and test.
        samples = transforms.Normalize((72.9403,), (90.0212,))(samples)
        sample_resource = self._torch_tensor_to_streamed_resource(preprocessed_identifier, samples)
        return sample_resource

    def _preprocess_target_resource(self, targets: torch.Tensor, preprocessed_identifier: str) -> StreamedResource:
        target_resource = self._torch_tensor_to_streamed_resource(preprocessed_identifier, targets)
        return target_resource
