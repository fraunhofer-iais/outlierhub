from typing import Tuple, Dict, List
from data_stack.io.resources import ResourceFactory, StreamedResource
from data_stack.io.storage_connectors import StorageConnector
from nltk.corpus import reuters
from collections import defaultdict
from flair.embeddings import DocumentPoolEmbeddings, WordEmbeddings
from flair.data import Sentence
import tempfile
import h5py
import os
import tqdm


class ReutersPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector
        self.split_names = ["train", "test"]

    def preprocess(self, dataset_identifier: str) -> StreamedResource:
        with tempfile.TemporaryFile() as temp_file:
            with h5py.File(temp_file, 'w') as h5py_file:
                for split_name in self.split_names:
                    self._preprocess_split(h5py_file, split_name)
                h5py_file.flush()
                temp_file.flush()
                streamed_resource = ResourceFactory.get_resource(dataset_identifier, temp_file)
                self.storage_connector.set_resource(dataset_identifier, streamed_resource)
                streamed_resource = self.storage_connector.get_resource(dataset_identifier)
        return streamed_resource

    def _preprocess_split(self, h5py_file: h5py.File, split_name: str) -> StreamedResource:
        split_raw, split_targets = self._get_raw_dataset_split(split_name)
        embedder_instances = [WordEmbeddings("glove")]
        doc_embedders = DocumentPoolEmbeddings(embedder_instances)
        sample_location = os.path.join(split_name, "samples")
        target_location = os.path.join(split_name, "targets")
        embedding_size = sum([embedder.embedding_length for embedder in embedder_instances])
        string_datatype = h5py.string_dtype(encoding='ascii')
        sample_dset = h5py_file.create_dataset(sample_location, shape=(len(split_raw), embedding_size,))
        target_dset = h5py_file.create_dataset(target_location, (len(split_targets),), dtype=string_datatype)
        for i in tqdm.tqdm(range(len(split_raw)), desc=f"Embedding {split_name} split"):
            embedded_doc = self._embed_document(split_raw[i], doc_embedders)
            sample_dset[i] = embedded_doc
            target_dset[i] = split_targets[i]

    def _get_raw_dataset_split(self, split_name: str) -> Tuple[List[str], List[str]]:
        def _get_doc_id_to_labels_mapping(labels: List[str]) -> Dict[str, List[str]]:
            """
            Get document categories
            """
            doc_ids_to_labels = defaultdict(list)
            for label in labels:
                doc_ids = reuters.fileids(label)
                for doc_id in doc_ids:
                    doc_ids_to_labels[doc_id].append(label)

            return doc_ids_to_labels

        document_ids = reuters.fileids()
        # split train and test based on their names
        doc_ids = list(filter(lambda doc: doc.startswith(split_name), document_ids))
        doc_ids_to_labels = _get_doc_id_to_labels_mapping(reuters.categories())
        # filter documents that belong to only one category
        # ie. multi-class setting instead of multi-label setting
        split_doc_ids = [doc_id for doc_id in doc_ids if len(doc_ids_to_labels[doc_id]) == 1]
        # raw text
        split_raw = [reuters.raw(doc_id) for doc_id in split_doc_ids]
        # labels
        split_targets = [doc_ids_to_labels[doc_id][0] for doc_id in split_doc_ids]
        return split_raw, split_targets

    def _embed_document(self, document_text: str, doc_embeddings: DocumentPoolEmbeddings):
        sentence = Sentence(document_text)
        doc_embeddings.embed(sentence)
        return sentence.get_embedding().data.cpu().numpy()
