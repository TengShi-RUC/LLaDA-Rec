import math
import os
import time
from collections import OrderedDict, defaultdict
from logging import getLogger

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_scheduler

from genrec.evaluator import Evaluator
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import config_for_log, get_file_name, get_total_steps, log


class Trainer:
    """
    A class that handles the training process for a model.

    Args:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        tokenizer (AbstractTokenizer): The tokenizer used for tokenizing the data.

    Attributes:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        evaluator (Evaluator): The evaluator used for evaluating the model.
        logger (Logger): The logger used for logging training progress.
        project_dir (str): The directory path for saving tensorboard logs.
        accelerator (Accelerator): The accelerator used for distributed training
        saved_model_ckpt (str): The file path for saving the trained model checkpoint.

    Methods:
        fit(train_dataloader, val_dataloader): Trains the model using the provided training and validation dataloaders.
        evaluate(dataloader, split='test'): Evaluate the model on the given dataloader.
        end(): Ends the training process and releases any used resources.
    """

    def __init__(self, config: dict, model: AbstractModel,
                 tokenizer: AbstractTokenizer):
        self.config = config
        self.model = model
        self.accelerator = config['accelerator']
        self.evaluator = Evaluator(config, tokenizer)
        self.logger = getLogger()

        self.saved_model_ckpt = os.path.join(config['result_dir'],
                                             config['ckpt_dir'],
                                             f"{config['run_time']}.pth")

        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)

    def fit(self, train_dataloader, val_dataloader):
        """
        Trains the model using the provided training and validation dataloaders.

        Args:
            train_dataloader: The dataloader for training data.
            val_dataloader: The dataloader for validation data.
        """
        optimizer = AdamW(self.model.parameters(),
                          lr=self.config['lr'],
                          weight_decay=self.config['weight_decay'])

        total_n_steps = get_total_steps(self.config, train_dataloader)
        if total_n_steps == 0:
            self.log('No training steps needed.')
            return None, None

        warmup_steps = math.floor(total_n_steps * self.config['warmup_ratio'])
        self.log(f"Total steps: {total_n_steps}, warmup steps: {warmup_steps}")

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_n_steps,
        )

        self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, scheduler)
        self.accelerator.init_trackers(
            project_name=get_file_name(self.config, suffix=''),
            config=config_for_log(self.config),
            init_kwargs={"tensorboard": {
                "flush_secs": 60
            }},
        )

        n_epochs = np.ceil(total_n_steps /
                           (len(train_dataloader) *
                            self.accelerator.num_processes)).astype(int)
        best_epoch = 0
        best_val_score = -1

        for epoch in range(n_epochs):
            start_time = time.time()
            self.model.train()

            total_loss = 0.0
            total_his_mask_loss = 0.0
            total_target_mask_loss = 0.0

            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
                disable=not self.accelerator.is_main_process,
            )
            for batch in train_progress_bar:
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(),
                                    self.config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.item()

                total_his_mask_loss += outputs.his_mask_loss.item()
                total_target_mask_loss += outputs.target_mask_loss.item()

            self.accelerator.log(
                {"Loss/train_loss": total_loss / len(train_dataloader)},
                step=epoch + 1)

            self.log(
                "[Epoch {}] Train Loss: {:.4f} His_Mask Loss: {:.4f} Target_Mask Loss: {:.4f} Running Time: {:.2f}s  lr: {:.6f}"
                .format(epoch + 1, total_loss / len(train_dataloader),
                        total_his_mask_loss / len(train_dataloader),
                        total_target_mask_loss / len(train_dataloader),
                        time.time() - start_time,
                        scheduler.get_last_lr()[0]))

            # Evaluation
            if (epoch + 1) % self.config['eval_interval'] == 0:

                all_results = self.evaluate(val_dataloader,
                                            split='val',
                                            epoch=epoch)

                if self.accelerator.is_main_process:
                    for key in all_results:
                        self.accelerator.log(
                            {f"Val_Metric/{key}": all_results[key]},
                            step=epoch + 1)
                    self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')

                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        self.save_model()
                        self.log(
                            f'[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}'
                        )

                if self.config[
                        'patience'] is not None and epoch + 1 - best_epoch >= self.config[
                            'patience']:
                    self.log(f'Early stopping at epoch {epoch + 1}')
                    break
        self.log(f'Best epoch: {best_epoch}, Best val score: {best_val_score}')

        return best_epoch, best_val_score

    def save_model(self, ):
        if self.config['use_ddp']:  # unwrap model for saving
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
        else:
            torch.save(self.model.state_dict(), self.saved_model_ckpt)

    def evaluate(self, dataloader, split='test', epoch=-1):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
            disable=not self.accelerator.is_main_process,
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {
                    k: v.to(self.accelerator.device)
                    for k, v in batch.items()
                }
                batch['split'] = split
                batch['epoch'] = epoch

                if self.config[
                        'use_ddp']:  # ddp, gather data from all devices for evaluation

                    if split == 'val':
                        preds = self.model.module.generate(
                            batch, n_return_sequences=self.evaluator.maxk_eval)
                    else:
                        preds = self.model.module.generate(
                            batch, n_return_sequences=self.evaluator.maxk)

                    all_preds, all_labels = self.accelerator.gather_for_metrics(
                        (preds, batch['labels']))

                    results = self.evaluator.calculate_metrics(
                        all_preds, all_labels, split)
                else:
                    if split == 'val':
                        preds = self.model.generate(
                            batch, n_return_sequences=self.evaluator.maxk_eval)
                    else:
                        preds = self.model.generate(
                            batch, n_return_sequences=self.evaluator.maxk)

                    results = self.evaluator.calculate_metrics(
                        preds, batch['labels'], split)

                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for metric in self.config['metrics']:

            topk_list = self.config['topk'] if (
                split == 'test') else self.config['val_topk']
            # for k in self.config['topk']:
            for k in topk_list:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()

        return output_results

    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()

    def log(self, message, level='info'):
        return log(message,
                   self.config['accelerator'],
                   self.logger,
                   level=level)
