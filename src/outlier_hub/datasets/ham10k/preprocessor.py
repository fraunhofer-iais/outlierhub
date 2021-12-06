from data_stack.io.storage_connectors import StorageConnector
import zipfile as ZipFile

class HAMPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector

    def preprocess(self,
                   raw_samples_identifier: str,
                   raw_targets_identifier: str,
                   prep_dataset_identifier: str):
     
        # +raw_samples_identifier = self.unzip_samples(raw_samples_identifier)
        print(type(raw_samples_identifier))

        raw_samples_identifier = raw_samples_identifier.split(',')
        print(type(raw_samples_identifier))
        print(raw_samples_identifier.__len__())
        print(raw_samples_identifier[0])
        print(raw_samples_identifier[1])

        # if not os.path.exists(prep_dataset_identifier):
        #     os.makedirs(os.path.dirname(prep_dataset_identifier), exist_ok=True)
        
        # with h5py.File(prep_dataset_identifier, 'w') as h5py_file:
        #    self.prepare_dataset(h5py_file,
        #                            raw_samples_identifier,
        #                            raw_targets_identifier,
        #                             )
            #h5py_file.flush()
            #temp_file.flush()
            #streamed_resource = ResourceFactory.get_resource(prep_dataset_identifier, temp_file)
            #self.storage_connector.set_resource(prep_dataset_identifier, streamed_resource)
            #streamed_resource = self.storage_connector.get_resource(prep_dataset_identifier)
        #return streamed_resource