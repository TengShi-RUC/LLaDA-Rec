import argparse
import importlib
import os
from logging import getLogger
from typing import Dict

import yaml
from accelerate import Accelerator

from genrec.utils import (get_config, get_dataset, getLogger, init_device,
                          init_logger, init_seed, parse_command_line_args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default='TokenModel',
                        help='Model name')
    parser.add_argument('--tokenizer_name',
                        type=str,
                        default='MultiHeadVQVAE',
                        help='Tokenizer name')
    parser.add_argument('--dataset',
                        type=str,
                        default='AmazonReviews2023',
                        help='Dataset name')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    config: Dict = get_config(model_name=args.model,
                              dataset_name=args.dataset,
                              config_file=None,
                              config_dict=command_line_configs)
    config['tokenizer_name'] = args.tokenizer_name
    config['device'], config['use_ddp'] = init_device()

    # Accelerator
    project_dir = os.path.join(config['tensorboard_log_dir'],
                               config["dataset"], config["model"])

    accelerator = Accelerator(log_with='tensorboard', project_dir=project_dir)
    config['accelerator'] = accelerator

    # Seed and Logger
    init_seed(config['rand_seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()

    for key, value in config.items():
        logger.info(f'{key}: {value}')

    tokenizer_name = config['tokenizer_name']
    try:
        tokenizer_class = getattr(
            importlib.import_module(
                f'genrec.tokenizers.{tokenizer_name}.tokenizer'),
            f'{tokenizer_name}Tokenizer')
    except:
        raise ValueError(f'Tokenizer "{tokenizer_name}" not found.')

    logger.info(f'[TOKENIZER] Loading tokenizer config')
    tokenizer_config: dict = yaml.safe_load(
        open(f'genrec/tokenizers/{tokenizer_name}/config.yaml', 'r'))

    for key in tokenizer_config.keys():
        if key in config.keys():
            tokenizer_config[key] = config[key]
        logger.info(f"{key}: {tokenizer_config[key]}")
    config.update(tokenizer_config)

    raw_dataset = get_dataset(args.dataset)(config)
    split_datasets = raw_dataset.split()
    logger.info(raw_dataset)

    tokenizer = tokenizer_class(config, raw_dataset)
    item2tokens = tokenizer.item2tokens
