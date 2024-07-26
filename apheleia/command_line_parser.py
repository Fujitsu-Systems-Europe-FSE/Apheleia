from apheleia.catalog import PipelinesCatalog, DatasetsCatalog, LossesCatalog, OptimizersCatalog, SchedulesCatalog

import os
import sys
import argparse


class CommandLineParser:
    def __init__(self, with_dataset, description=''):
        self._with_dataset = with_dataset
        self._subparsers = {}

        self._parser = argparse.ArgumentParser(description=description, fromfile_prefix_chars='@')
        self._action = self._parser.add_subparsers(dest='action')
        self._define_default_subparsers()

    def _define_default_subparsers(self):
        self._define_train_subparser()
        self._define_infer_subparser()

    def _define_train_subparser(self):
        train_parser = self.add_subparser('train')
        train_parser.add_argument('arch', type=str, choices=PipelinesCatalog().choices(), help='Model architectures.')
        if self._with_dataset:
            train_parser.add_argument('dataset_class', choices=DatasetsCatalog().choices(), type=str, help='training dataset type.')
            train_parser.add_argument('dataset', type=str, help='path of the train dataset.')
            train_parser.add_argument('--val-dataset', type=str, help='Path of the validation dataset. By default train dataset will be used.')

        # train_parser.add_argument('--test-dataset', type=str, help='Path of the test dataset. Disabled if no dataset is defined.')
        train_parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size (comma separated list for dynamic batch sizes).')
        train_parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=200, help='Learning epochs.')
        train_parser.add_argument('-o', '--output', dest='outdir', type=str, help='Model output directory. Default: timestamped folder in current directory.')
        train_parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory if it already exists.')
        train_parser.add_argument('--gpus', type=str, default='', help='GPUs id to use, for example 0,1, etc. -1 to use cpu. Default: use all GPUs.')
        train_parser.add_argument('--distributed', action='store_true', help='Enable distributed training among multiple nodes.')
        train_parser.add_argument('--rank', type=int, help='Please see pytorch documentation for distributed execution.')
        train_parser.add_argument('--world-size', type=int, help='Please see pytorch documentation for distributed execution.')
        train_parser.add_argument('--master', type=str, help='Please see pytorch documentation for distributed execution.')
        train_parser.add_argument('--runs', type=int, default=1, help='Number of training runs. Same arguments are re-used for all the runs.')
        train_parser.add_argument('--fp16', action='store_true', help='Enables Automated Mixed Precision for reduced memory consumption and faster computation. Requires Turing+ GPUs.')
        train_parser.add_argument('--loss-scaling', action='store_true', help='Enables loss scaling in FP16 mode. Requires Turing+ GPUs.')
        train_parser.add_argument('-d', '--daemon', action='store_true', help='Runs as daemon process. Logs saved in output directory.')
        train_parser.add_argument('-w', '--workers', dest='workers', type=int, default=os.cpu_count() // 2, help='Number of workers to use.')
        train_parser.add_argument('--target-metric', type=float, help='Early training stopping whn target metric is reached.')
        train_parser.add_argument('--carbon-footprint', action='store_true', help='Compute carbon footprint.')
        train_parser.add_argument('--loss', type=str, choices=LossesCatalog().choices(), help='Loss to use during training.')

        # Models pre-trained weights and reproducibility
        train_parser.add_argument('--seed', type=int, help='Set seed for reproducibility.')
        train_parser.add_argument('--models', type=str, help='Models path. Loads whole model pretrained weights.')
        train_parser.add_argument('--resume', action='store_true', help='Resume training by restoring model and optimizers states.')

        # Generic optimizers params
        train_parser.add_argument('--optimizers', type=str, nargs='+', choices=OptimizersCatalog().keys(), help='Optimizers to use.')
        train_parser.add_argument('--optimizers-params', type=str, nargs='+', help='E.g. <arg1>:<kwarg1>=<value> and are colon separated. Expected params for optimizer can be found in PyTorch documentation.')
        train_parser.add_argument('--lr-schedules', type=str, nargs='+', choices=SchedulesCatalog().keys(), help='Learning rate schedules to use.')
        train_parser.add_argument('--schedules-params', type=str, nargs='+', help='E.g. <arg1>:<kwarg1>=<value> and are colon separated. Expected params for a scheduler can be found in PyTorch documentation.')
        train_parser.add_argument('--clip-grad-norm', type=float, help='Clip gradients norm')
        train_parser.add_argument('--clip-grad-value', type=float, help='Clip gradients value')
        train_parser.add_argument('--num-accum', type=int, default=1, help='Accumulate gradients to simulate a larger batch size')

        # Trainer interval opts and logs options
        train_parser.add_argument('--wandb', action='store_true', help='Upload logs to Weight and Biases.')
        train_parser.add_argument('--sweep', type=str, help='Wandb sweep conf file for hyperparameters exploration. Enables --wandb')
        train_parser.add_argument('--log-interval', type=int, default=5, help='Iterations log interval.')
        train_parser.add_argument('--chkpt-interval', type=int, help='Model checkpointing interval (epochs).')
        train_parser.add_argument('--max-chkpts', type=int, default=5, help='Maximum number of checkpoints to keep. Only the best will be kept.')
        train_parser.add_argument('--val-interval', type=int, default=10, help='Model validation interval (epochs).')
        train_parser.add_argument('--test-interval', type=int, default=10, help='Model test interval (epochs).')
        train_parser.add_argument('--report-interval', type=int, default=2, help='Model report interval (epochs).')
        train_parser.add_argument('--thumb-interval', type=int, default=2, help='Thumbnail generation interval (epochs).')
        train_parser.add_argument('--stats-interval', type=int, default=10, help='Networks stats report interval (epochs).')

    def _define_infer_subparser(self):
        # TODO Avoid duplicated help text
        infer_parser = self.add_subparser('infer')
        infer_parser.add_argument('models', type=str, help='Models path. Loads whole model pretrained weights.')
        infer_parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size (comma separated list for dynamic batch sizes).')
        infer_parser.add_argument('--arch', type=str, choices=PipelinesCatalog().choices(), help='Model architectures.')
        infer_parser.add_argument('--gpus', type=str, default='', help='GPUs id to use, for example 0,1, etc. -1 to use cpu. Default: use all GPUs.')
        infer_parser.add_argument('--seed', type=int, help='Set seed for reproducibility.')
        infer_parser.add_argument('-o', '--output', dest='outdir', type=str, help='Model output directory. Default: timestamped folder in current directory.')
        infer_parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory if it already exists.')

    def add_subparser(self, name):
        if name in self._subparsers:
            raise Exception('Subparser already exists.')
        subparser = self._action.add_parser(name)
        self._subparsers[name] = subparser
        return subparser

    def get_subparser(self, name):
        return self._subparsers[name]

    def parse(self):
        if len(sys.argv) == 1:
            self._parser.print_help(sys.stderr)
            exit(1)

        return self._parser.parse_args()
