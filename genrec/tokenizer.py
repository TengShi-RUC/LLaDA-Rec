import os
from logging import getLogger

import numpy as np
from sentence_transformers import SentenceTransformer

from genrec.dataset import AbstractDataset


class AbstractTokenizer:

    def __init__(self, config: dict, dataset: AbstractDataset):
        self.config = config
        self.logger = getLogger()
        self.eos_token = None
        self.collate_fn = {'train': None, 'val': None, 'test': None}

    def _init_tokenizer(self):
        raise NotImplementedError('Tokenizer initialization not implemented.')

    def tokenize(self, datasets):
        raise NotImplementedError('Tokenization not implemented.')

    @property
    def vocab_size(self):
        raise NotImplementedError('Vocabulary size not implemented.')

    @property
    def padding_token(self):
        return 0

    @property
    def max_token_seq_len(self):
        raise NotImplementedError(
            'Maximum token sequence length not implemented.')

    def log(self, message, level='info'):
        from genrec.utils import log
        return log(message,
                   self.config['accelerator'],
                   self.logger,
                   level=level)


class SemIDTokenizer(AbstractTokenizer):

    def __init__(self, config: dict, dataset: AbstractDataset):
        super().__init__(config, dataset)

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """
        Encodes the sentence embeddings for the given dataset and saves them to the specified output path.

        Args:
            dataset (AbstractDataset): The dataset containing the sentences to encode.
            output_path (str): The path to save the encoded sentence embeddings.

        Returns:
            numpy.ndarray: The encoded sentence embeddings.
        """
        assert self.config['metadata'] == 'sentence', \
            'TIGERTokenizer only supports sentence metadata.'

        sent_emb_model = SentenceTransformer(self.config['sent_emb_model']).to(
            self.config['device'])

        meta_sentences = []  # 1-base, meta_sentences[0] -> item_id = 1
        for i in range(1, dataset.n_items):
            meta_sentences.append(
                dataset.item2meta[dataset.id_mapping['id2item'][i]])
        sent_embs = sent_emb_model.encode(
            meta_sentences,
            convert_to_numpy=True,
            batch_size=self.config['sent_emb_batch_size'],
            show_progress_bar=True,
            device=self.config['device'])

        sent_embs.tofile(output_path)
        return sent_embs

    def load_sent_emb(self, dataset: AbstractDataset):
        """
        Loads the sentence embeddings for the given dataset from the specified path.

        Args:
            dataset (AbstractDataset): The dataset containing the sentences to encode.

        Returns:
            numpy.ndarray: The loaded sentence embeddings.
        """
        assert self.config['metadata'] == 'sentence', \
            'TIGERTokenizer only supports sentence metadata.'

        # Load or encode sentence embeddings
        sent_emb_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb')
        if os.path.exists(sent_emb_path):
            self.log(
                f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...'
            )
            sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(
                -1, self.config['sent_emb_dim'])
        else:
            self.log(f'[TOKENIZER] Encoding sentence embeddings...')
            sent_embs = self._encode_sent_emb(dataset, sent_emb_path)

        # PCA
        if self.config['sent_emb_pca'] > 0:
            self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
            sent_embs = pca.fit_transform(sent_embs)

        return sent_embs

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """
        Get a boolean mask indicating which items are used for training.

        Args:
            dataset (AbstractDataset): The dataset containing the item sequences.

        Returns:
            np.ndarray: A boolean mask indicating which items are used for training.
        """
        items_for_training = set()
        for item_seq in dataset.split_data['train']['item_seq']:
            for item in item_seq:
                items_for_training.add(item)
        self.log(
            f'[TOKENIZER] Items for training: {len(items_for_training)} of {dataset.n_items - 1}'
        )
        mask = np.zeros(dataset.n_items - 1, dtype=bool)
        for item in items_for_training:
            mask[dataset.item2id[item] - 1] = True
        return mask
