import importlib

import yaml

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class LLaDARecTokenizer(AbstractTokenizer):

    def __init__(self, config: dict, dataset: AbstractDataset):
        super(LLaDARecTokenizer, self).__init__(config, dataset)
        self.accelerator = config['accelerator']

        tokenizer_name = self.config['tokenizer_name']
        try:
            tokenizer_class = getattr(
                importlib.import_module(
                    f'genrec.tokenizers.{tokenizer_name}.tokenizer'),
                f'{tokenizer_name}Tokenizer')
        except:
            raise ValueError(f'Tokenizer "{tokenizer_name}" not found.')

        self.log(f'[TOKENIZER] Loading tokenizer config')
        tokenizer_config: dict = yaml.safe_load(
            open(f'genrec/tokenizers/{tokenizer_name}/config.yaml', 'r'))

        for key in tokenizer_config.keys():
            if key in config.keys():
                tokenizer_config[key] = config[key]
            self.log(f"{key}: {tokenizer_config[key]}")
        config.update(tokenizer_config)
        self.config = config

        self.tokenizer = tokenizer_class(config, dataset)

        self.item2id = dataset.item2id
        self.item2tokens = self.tokenizer.item2tokens
        self.mask_token = self.tokenizer.eos_token + 1
        self.ignored_label = self.tokenizer.ignored_label

    @property
    def n_digit(self):
        return self.tokenizer.n_digit

    @property
    def codebook_sizes(self):
        return self.tokenizer.codebook_sizes

    @property
    def max_token_seq_len(self) -> int:
        return self.config['max_item_seq_len']

    @property
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size for the TIGER tokenizer.
        """
        return self.mask_token + 1

    def _tokenize_items(self, item_seq: list, test=False):
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids = [0] * pad_lens + input_ids
        attention_mask = [0] * pad_lens + attention_mask

        if test:
            labels = list(self.item2tokens[item_seq[-1]])
        else:
            labels = [self.item2id[item_seq[-1]]]

        return input_ids, attention_mask, labels, seq_lens

    def tokenize_function(self, example: dict, split: str) -> dict:
        max_item_seq_len = self.config['max_item_seq_len']
        item_seq = example['item_seq'][0]

        if split == 'train':
            all_input_ids, all_attention_mask, all_labels, all_seq_lens= [],[],[],[]
            for i in range(2, len(item_seq) + 1):
                cur_item_seq = item_seq[max(0, i - max_item_seq_len - 1):i]
                input_ids, attention_mask, labels, seq_lens = self._tokenize_items(
                    cur_item_seq)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)
            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels,
                'seq_lens': all_seq_lens,
            }

        else:
            input_ids, attention_mask, labels, seq_lens = self._tokenize_items(
                item_seq[-(max_item_seq_len + 1):], test=True)
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'labels': [labels],
                'seq_lens': [seq_lens]
            }

    def tokenize(self, datasets: dict) -> dict:
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: ')

        for split in datasets:
            self.log(
                f"Tokenized {split} set: {len(tokenized_datasets[split])}")
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets
