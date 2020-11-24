from data_stack.io.resources import ResourceFactory, StreamedResource
from data_stack.io.storage_connectors import StorageConnector
from flair.embeddings import DocumentPoolEmbeddings, WordEmbeddings
from flair.data import Sentence
import tempfile
import h5py
import os
import tqdm
from typing import Dict
import pandas as pd


class AtisPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector

    def preprocess(self, preprocessed_dataset_identifier: str, raw_train_identifier: str, raw_val_identifier: str, raw_test_identifier: str) -> StreamedResource:
        split_name_to_identifier = {"train": raw_train_identifier, "val": raw_val_identifier, "test": raw_test_identifier}

        with tempfile.TemporaryFile() as temp_file:
            with h5py.File(temp_file, 'w') as h5py_file:
                self._preprocess_splits(h5py_file, split_name_to_identifier)
                h5py_file.flush()
                temp_file.flush()
                streamed_resource = ResourceFactory.get_resource(preprocessed_dataset_identifier, temp_file)
                self.storage_connector.set_resource(preprocessed_dataset_identifier, streamed_resource)
                streamed_resource = self.storage_connector.get_resource(preprocessed_dataset_identifier)
        return streamed_resource

    def _preprocess_splits(self, h5py_file: h5py.File, split_name_to_identifier: Dict[str, str]):
        for split_name, split_identifier in split_name_to_identifier.items():
            with self.storage_connector.get_resource(split_identifier) as raw_split_resource:
                self._preprocess_split(h5py_file, split_name=split_name, raw_split_resource=raw_split_resource)

    def _preprocess_split(self, h5py_file: h5py.File, split_name: str, raw_split_resource: StreamedResource) -> StreamedResource:
        split_df = pd.read_csv(raw_split_resource, delimiter=",")
        targets_df = split_df["intent"]
        sample_texts_df = split_df["tokens"]

        embedder_instances = [WordEmbeddings("en-glove")]
        doc_embedders = DocumentPoolEmbeddings(embedder_instances)

        sample_location = os.path.join(split_name, "samples")
        target_location = os.path.join(split_name, "targets")
        embedding_size = sum([embedder.embedding_length for embedder in embedder_instances])
        string_datatype = h5py.string_dtype(encoding='ascii')
        sample_dset = h5py_file.create_dataset(sample_location, shape=(len(split_df), embedding_size,))
        target_dset = h5py_file.create_dataset(target_location, (len(split_df),), dtype=string_datatype)
        for i in tqdm.tqdm(range(len(split_df)), desc=f"Embedding {split_name} split"):
            doc_text = AtisPreprocessor._clean_text(sample_texts_df.iloc[i])
            embedded_doc = self._embed_document(doc_text, doc_embedders)
            sample_dset[i] = embedded_doc
            target_dset[i] = targets_df.iloc[i]

    @staticmethod
    def _clean_text(text: str):
        # text
        text = text.replace("BOS ", "")
        text = text.replace(" EOS", "")
        return text

    def _embed_document(self, document_text: str, doc_embeddings: DocumentPoolEmbeddings):
        sentence = Sentence(document_text)
        doc_embeddings.embed(sentence)
        return sentence.get_embedding().data.cpu().numpy()
