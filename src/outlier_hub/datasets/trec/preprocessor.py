from data_stack.io.resources import ResourceFactory, StreamedResource
from data_stack.io.storage_connectors import StorageConnector
from flair.embeddings import DocumentPoolEmbeddings, WordEmbeddings
from flair.data import Sentence
import tempfile
import h5py
import os
import tqdm
from torchtext.legacy import datasets, data


class TrecPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector
        self.split_names = ["train", "test"]

    def preprocess(self, preprocessed_dataset_identifier: str) -> StreamedResource:
        with tempfile.TemporaryFile() as temp_file:
            with h5py.File(temp_file, 'w') as h5py_file:
                self._preprocess_splits(h5py_file)
                h5py_file.flush()
                temp_file.flush()
                streamed_resource = ResourceFactory.get_resource(preprocessed_dataset_identifier, temp_file)
                self.storage_connector.set_resource(preprocessed_dataset_identifier, streamed_resource)
                streamed_resource = self.storage_connector.get_resource(preprocessed_dataset_identifier)
        return streamed_resource

    def _preprocess_splits(self, h5py_file: h5py.File):
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
        LABEL = data.Field(sequential=False)
        with tempfile.TemporaryDirectory() as tmpdirname:
            train_set, test_set = datasets.trec.TREC.splits(TEXT, LABEL, root=tmpdirname, fine_grained=True)
            self._preprocess_split(h5py_file, split_name="train", dataset_split=train_set)
            self._preprocess_split(h5py_file, split_name="test", dataset_split=test_set)

    def _preprocess_split(self, h5py_file: h5py.File, split_name: str, dataset_split) -> StreamedResource:
        embedder_instances = [WordEmbeddings("en-glove")]
        doc_embedders = DocumentPoolEmbeddings(embedder_instances)
        sample_location = os.path.join(split_name, "samples")
        target_location = os.path.join(split_name, "targets")
        embedding_size = sum([embedder.embedding_length for embedder in embedder_instances])
        string_datatype = h5py.string_dtype(encoding='ascii')
        sample_dset = h5py_file.create_dataset(sample_location, shape=(len(dataset_split), embedding_size,))
        target_dset = h5py_file.create_dataset(target_location, (len(dataset_split),), dtype=string_datatype)
        for i in tqdm.tqdm(range(len(dataset_split)), desc=f"Embedding {split_name} split"):
            doc_text = " ".join(dataset_split[i].text)
            embedded_doc = self._embed_document(doc_text, doc_embedders)
            sample_dset[i] = embedded_doc
            target_dset[i] = dataset_split[i].label

    def _embed_document(self, document_text: str, doc_embeddings: DocumentPoolEmbeddings):
        sentence = Sentence(document_text)
        doc_embeddings.embed(sentence)
        return sentence.get_embedding().data.cpu().numpy()
