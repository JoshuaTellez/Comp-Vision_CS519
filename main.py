# Taken from original 3DR2N2 repository

# Modules used
import numpy as np
import argparse
import pprint
import logging
import multiprocessing as mp

# Theano
import theano.sandbox.cuda

# Library from repo
from lib.config import cfg, cfg_from_file, cfg_from_list

# My reimplementations
from my_3DR2N2.my_train_net import train_net

# Parse the arguments from command line
def parse_args():
    parser = argparse.ArgumentParser(description='Main 3Deverything train/test file')
    
    # Add which gpu to use
    parser.add_argument(
        '--gpu',
        dest='gpu_id',
        help='GPU device id to use [gpu0]',
        default=cfg.CONST.DEVICE,
        type=str)
    
    # Add configuration flag
    parser.add_argument(
        '--cfg',
        dest='cfg_files',
        action='append',
        help='optional config file',
        default=None,
        type=str)
    
    # Add random seed
    parser.add_argument(
        '--rand', dest='randomize', help='randomize (do not use a fixed seed)', action='store_true')
    # Add if testing will be done
    parser.add_argument(
        '--test', dest='test', help='randomize (do not use a fixed seed)', action='store_true')
    # Add which kind of network to be used
    parser.add_argument('--net', dest='net_name', help='name of the net', default=None, type=str)
    
    # Add the name of network model
    parser.add_argument(
        '--model', dest='model_name', help='name of the network model', default=None, type=str)
    # Add the size of batch
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        help='name of the net',
        default=cfg.CONST.BATCH_SIZE,
        type=int)
    
    # Add number of training iterations
    parser.add_argument(
        '--iter',
        dest='iter',
        help='number of iterations',
        default=cfg.TRAIN.NUM_ITERATION,
        type=int)
    
    # Add where to find dataset
    parser.add_argument(
        '--dataset', dest='dataset', help='dataset config file', default=None, type=str)
    parser.add_argument(
        '--set', dest='set_cfgs', help='set config keys', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--exp', dest='exp', help='name of the experiment', default=None, type=str)
    
    # Add what to name and where to place weights
    parser.add_argument(
        '--weights', dest='weights', help='Initialize network from the weights file', default=None)
    parser.add_argument('--out', dest='out_path', help='set output path', default=cfg.DIR.OUT_PATH)
    
    # Start training from this iteration
    parser.add_argument(
        '--init-iter',
        dest='init_iter',
        help='Start from the specified iteration',
        default=cfg.TRAIN.INITIAL_ITERATION)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print('Called with args:')
    print(args)

    # Set main gpu
    theano.sandbox.cuda.use(args.gpu_id)

    # Initialize config files
    if args.cfg_files is not None:
        for cfg_file in args.cfg_files:
            cfg_from_file(cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if not args.randomize:
        np.random.seed(cfg.CONST.RNG_SEED)
    
    # Initialize config variables pertaining to training
    if args.batch_size is not None:
        cfg_from_list(['CONST.BATCH_SIZE', args.batch_size])
    if args.iter is not None:
        cfg_from_list(['TRAIN.NUM_ITERATION', args.iter])
    if args.net_name is not None:
        cfg_from_list(['NET_NAME', args.net_name])
    if args.model_name is not None:
        cfg_from_list(['CONST.NETWORK_CLASS', args.model_name])
    if args.dataset is not None:
        cfg_from_list(['DATASET', args.dataset])
    if args.exp is not None:
        cfg_from_list(['TEST.EXP_NAME', args.exp])
    if args.out_path is not None:
        cfg_from_list(['DIR.OUT_PATH', args.out_path])
    if args.weights is not None:
        cfg_from_list(['CONST.WEIGHTS', args.weights, 'TRAIN.RESUME_TRAIN', True,
                       'TRAIN.INITIAL_ITERATION', int(args.init_iter)])

    print('Using config:')
    pprint.pprint(cfg)

    if not args.test:
        train_net()


if __name__ == '__main__':
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)
    main()
