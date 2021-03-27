from typing import List
from data_stack.io.resources import ResourceFactory, StreamedResource
from data_stack.io.storage_connectors import StorageConnector
from flair.embeddings import DocumentPoolEmbeddings, WordEmbeddings
from flair.data import Sentence
import tempfile
import h5py
import tqdm
from sklearn.datasets import fetch_20newsgroups


class NewsGroupsPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector

    def preprocess(self, dataset_identifier: str) -> StreamedResource:
        with tempfile.TemporaryDirectory() as tmp_raw_dataset:
            raw_data = fetch_20newsgroups(data_home=tmp_raw_dataset, subset="all")
            samples: List[str] = raw_data.data
            targets: List[int] = raw_data.target
            target_names: List[str] = raw_data.target_names

        with tempfile.TemporaryFile() as temp_file:
            with h5py.File(temp_file, 'w') as h5py_file:
                self._preprocess_split(h5py_file, samples, targets, target_names)
                h5py_file.flush()
                temp_file.flush()
                streamed_resource = ResourceFactory.get_resource(dataset_identifier, temp_file)
                self.storage_connector.set_resource(dataset_identifier, streamed_resource)
                streamed_resource = self.storage_connector.get_resource(dataset_identifier)
        return streamed_resource

    def _preprocess_split(self, h5py_file: h5py.File, samples: List[str], targets: List[int], target_names: List[str]) -> StreamedResource:
        embedder_instances = [WordEmbeddings("glove")]
        doc_embedders = DocumentPoolEmbeddings(embedder_instances)
        sample_location = "samples"
        target_location = "targets"
        embedding_size = sum([embedder.embedding_length for embedder in embedder_instances])
        string_datatype = h5py.string_dtype(encoding='ascii')
        sample_dset = h5py_file.create_dataset(sample_location, shape=(len(samples), embedding_size,))
        target_dset = h5py_file.create_dataset(target_location, (len(targets),), dtype=string_datatype)
        for i in tqdm.tqdm(range(len(samples)), desc="Embedding full split"):
            embedded_doc = self._embed_document(samples[i], doc_embedders)
            sample_dset[i] = embedded_doc
            target_dset[i] = target_names[targets[i]]

    def _embed_document(self, document_text: str, doc_embeddings: DocumentPoolEmbeddings):
        sentence = Sentence(document_text)
        doc_embeddings.embed(sentence)
        return sentence.get_embedding().data.cpu().numpy()
