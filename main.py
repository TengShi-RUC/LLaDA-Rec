import argparse
import os

from genrec.utils import get_pipeline, parse_command_line_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default='LLaDARec',
                        help='Model name')
    parser.add_argument('--dataset',
                        type=str,
                        default='AmazonReviews2023',
                        help='Dataset name')
    return parser.parse_known_args()


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    pipeline = get_pipeline(args.model)(model_name=args.model,
                                        dataset_name=args.dataset,
                                        config_dict=command_line_configs)
    pipeline.run()
