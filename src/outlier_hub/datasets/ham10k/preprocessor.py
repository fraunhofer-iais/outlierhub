import os 
from data_stack.io.storage_connectors import StorageConnector
from zipfile import ZipFile
import logging

class HAMPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector

    def unzip_samples(self, sample_file: str):

        new_filepath = sample_file.split('.')[1]

        zip_file = ZipFile(sample_file, 'r')

        zip_infos = zip_file.infolist()

        for zipinfo in zip_infos:
            archive_filename = zipinfo.filename

            zipinfo.filename = os.path.basename( zipinfo.filename)

            if not zipinfo.filename == '' :
                zip_file.extract(member = archive_filename, path = new_filepath)

        zip_file.close()

        os.remove(sample_file)

        return new_filepath

    def preprocess(self,
                   raw_samples_identifier: str,
                   raw_targets_identifier: str,
                   prep_dataset_identifier: str):
     
        raw_samples_identifier = self.unzip_samples(raw_samples_identifier)
        print(type(raw_samples_identifier))
        print(len(raw_samples_identifier))
        # logging.debug(f'samples = zipfile.ZipFile(raw_samples_identifier, 'r')')
        # samples = zipfile.ZipFile(raw_samples_identifier, 'r')
        
        # print(type(samples))
        # print(samples)

        # samples = raw_samples_identifier.split(',')
        # logging.debug(f'samples = raw_samples_identifier.split(',')')
        # print(type(samples))
        # print(samples)


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