import argparse
import os
import vizdoom
from src.utils import get_experiment_path
from src.logger import get_logger
from src.args import parse_game_args


parser = argparse.ArgumentParser(description='Barney')
parser.add_argument("--main_experiment_path", type=str, default="./experiment",  help="Experiment Path")               
parser.add_argument("--exp_name", type=str, default="default", help="Experiment name")
args, remaining = parser.parse_known_args()
assert len(args.exp_name.strip()) > 0

# create a directory for the experiment / create a logger
experiment_path    = get_experiment_path(args.main_experiment_path, args.exp_name)
logger = get_logger(filepath=os.path.join(experiment_path, 'train.log'))
logger.info('========== Running DOOM ==========')
logger.info('Experiment will be saved in: %s'  % experiment_path)

# load DOOM
parse_game_args(remaining + ['--experiment_path', experiment_path])
