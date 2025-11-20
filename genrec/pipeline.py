import copy
import json
import os
from logging import getLogger
from typing import Dict, Union

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import (get_config, get_dataset, get_model, get_tokenizer,
                          get_trainer, init_device, init_logger, init_seed,
                          log)


class Pipeline:

    def __init__(
        self,
        model_name: Union[str, AbstractModel],
        dataset_name: Union[str, AbstractDataset],
        tokenizer: AbstractTokenizer = None,
        trainer=None,
        config_dict: dict = None,
        config_file: str = None,
    ):
        self.config = get_config(model_name=model_name,
                                 dataset_name=dataset_name,
                                 config_file=config_file,
                                 config_dict=config_dict)
        # Automatically set devices and ddp
        self.config['device'], self.config['use_ddp'] = init_device()

        # Accelerator
        self.project_dir = os.path.join(self.config['tensorboard_log_dir'],
                                        self.config["dataset"],
                                        self.config["model"])

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(log_with='tensorboard',
                                       project_dir=self.project_dir,
                                       kwargs_handlers=[ddp_kwargs])
        self.config['accelerator'] = self.accelerator

        # Seed and Logger
        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()

        for key, value in self.config.items():
            self.log(f'{key}: {value}')

        # Dataset
        self.raw_dataset = get_dataset(dataset_name)(self.config)
        self.log(self.raw_dataset)
        self.split_datasets = self.raw_dataset.split()

        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer(self.config, self.raw_dataset)
        else:
            assert isinstance(
                model_name, str
            ), 'Tokenizer must be provided if model_name is not a string.'
            self.tokenizer = get_tokenizer(model_name)(self.config,
                                                       self.raw_dataset)
        self.tokenized_datasets = self.tokenizer.tokenize(self.split_datasets)

        for key, value in self.tokenized_datasets.items():
            self.log(f'{key} dataset size: {len(value)}')

        # Model
        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config, self.raw_dataset,
                                               self.tokenizer)

        self.log(self.model)
        self.log(self.model.n_parameters)

        # Trainer
        if trainer is not None:
            self.trainer = trainer
        else:
            self.trainer = get_trainer(model_name)(self.config, self.model,
                                                   self.tokenizer)

    def run(self):
        # DataLoader
        train_dataloader = DataLoader(
            self.tokenized_datasets['train'],
            batch_size=self.config['train_batch_size'],
            shuffle=True,
            collate_fn=self.tokenizer.collate_fn['train'])
        val_dataloader = DataLoader(
            self.tokenized_datasets['val'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['val'])
        test_dataloader = DataLoader(
            self.tokenized_datasets['test'],
            batch_size=self.config['test_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test'])

        best_epoch, best_val_score = self.trainer.fit(train_dataloader,
                                                      val_dataloader)

        self.accelerator.wait_for_everyone()

        self.model = self.accelerator.unwrap_model(self.model)
        self.log(
            f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}'
        )
        self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))

        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader)

        test_results = self.trainer.evaluate(test_dataloader)

        if self.accelerator.is_main_process:
            for key in test_results:
                self.accelerator.log({f'Test_Metric/{key}': test_results[key]})
        self.log(f'Test Results: {test_results}')

        results: Dict = copy.deepcopy(self.config)
        for key, value in results.items():
            try:
                results[key] = str(value)
            except:
                pass

        results.update({
            'best_epoch': best_epoch,
            'best_val_score': best_val_score,
            'test_results': test_results,
        })

        result_path = os.path.join(self.config['result_dir'], 'results',
                                   f"{self.config['run_time']}.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=4)

        self.trainer.end()
        return {
            'best_epoch': best_epoch,
            'best_val_score': best_val_score,
            'test_results': test_results,
        }

    def log(self, message, level='info'):
        return log(message,
                   self.config['accelerator'],
                   self.logger,
                   level=level)
