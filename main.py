#!/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name: entrance.py  
 Author: MASA (Refactored)
 Description: Entry point for the MASA framework
--------------------------------
'''

import os
import random
import numpy as np
import torch as th
import datetime
import time
import argparse
from config import Config
from runners import run_rl_only, run_rl_controller

# Configure CUDA and PyTorch for deterministic execution
def setup_environment():
    """Set up the execution environment for deterministic results"""
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    th.use_deterministic_algorithms(True)

def seed_everything(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MASA Framework')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the YAML configuration file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (overrides timestamp-based seed)')
    return parser.parse_args()

def entrance():
    """
    Entrance function for running the MASA framework
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup environment
    setup_environment()
    
    # Generate seed based on current timestamp if not provided
    current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if args.seed is not None:
        rand_seed = args.seed
    else:
        rand_seed = int(time.mktime(datetime.datetime.strptime(current_date, '%Y-%m-%d-%H-%M-%S').timetuple()))
    
    # Set all random seeds
    seed_everything(rand_seed)

    # Track execution time
    start_cputime = time.process_time()
    start_systime = time.perf_counter()
    
    # Initialize configuration
    config = Config(config_path=args.config, seed_num=rand_seed, current_date=current_date) 
    config.print_config()
    
    # Run the appropriate function based on mode
    if config.mode == 'RLonly':
        run_rl_only(config)
    elif config.mode == 'RLcontroller':
        run_rl_controller(config)
    elif config.mode == 'Benchmark':
        raise NotImplementedError("Please refer to the corresponding papers and their provided implementations.")
    else:
        raise ValueError(f'Unexpected mode {config.mode}')

    # Report total execution time
    end_cputime = time.process_time()
    end_systime = time.perf_counter()
    print(f"[Done] Total cputime: {np.round(end_cputime - start_cputime, 2)} s, "
          f"system time: {np.round(end_systime - start_systime, 2)} s")
    
def main():
    entrance()

if __name__ == '__main__':
    main()