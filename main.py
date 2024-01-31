"""

Author:
 
Ahmed Telili
 
Date Created:
 
29 01 2024
 
"""
import argparse
import pathlib
import yaml
import torch.nn as nn
from src.dataset import data_builder
from src.learner.learner import StandardLearner
from src.utils import common_util,  plugin


def parse_arguments():
    """Return arguments for VSR."""
    parser = argparse.ArgumentParser(description='Codebase for VSR.')
    parser.add_argument(
        '--process',
        help='Process type.',
        type=str,
        default='train',
        choices=['train', 'test'],
        required=True
    )
    parser.add_argument(
        '--config_path',
        help='Path of yaml config file of the application.',
        type=str,
        default=None,
        required=True
    )

    return parser.parse_args()

def main(args):
    """Run main function for vsr experiments.

    Args:
        args: A `dict` contain augments 'process' and 'config_path'.
    """
    # Prepare configurations
    with open(args.config_path, 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    log_dir = config.pop('log_dir')

    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)  # Ensure log_dir exists
    # Prepare dataset
    dataset = data_builder.build_dataset(config['dataset'], config['model'])
    # Prepare model
    model_builder = plugin.plugin_from_file(
    config['model']['path'], config['model']['name'], nn.Module)

    model = model_builder(config['model']['scale_factor'])

    # Copy model file to log directory
    common_util.copy_file(config['model']['path'], log_dir)

    # Prepare learner
    learner = StandardLearner(config, model, dataset, log_dir)

    if args.process == 'train':
        learner.train()
    elif args.process == 'test':
        learner.test()
    else:
        raise ValueError(f'Wrong process type {args.process}')

if __name__ == '__main__':

    arguments = parse_arguments()
    main(arguments)