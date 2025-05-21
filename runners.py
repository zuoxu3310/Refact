#!/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name: runners.py  
 Author: MASA (Refactored)
 Description: Implementation of different running modes for the MASA framework
--------------------------------
'''

import os
import numpy as np
import pandas as pd
import time
import timeit
from typing import Dict, Optional, Any

from utils.featGen import FeatureProcesser
from utils.tradeEnv import StockPortfolioEnv, StockPortfolioEnv_cash
from utils.model_pool import model_select, benchmark_algo_select
from utils.callback_func import PoCallback
from RL_controller.market_obs import MarketObserver, MarketObserver_Algorithmic

def load_and_process_data(config):
    """
    Load and preprocess dataset
    
    Args:
        config: Configuration object
    
    Returns:
        tuple: (data_dict, tech_indicator_lst, stock_num)
    """
    # Get dataset
    mkt_name = config.market_name
    fpath = os.path.join(config.dataDir, f'{mkt_name}_{config.topK}_{config.freq}.csv')
    if not os.path.exists(fpath):
        raise ValueError(f"Cannot load the data from {fpath}")
    data = pd.DataFrame(pd.read_csv(fpath, header=0))
    
    # Preprocess features
    featProc = FeatureProcesser(config=config)
    data_dict = featProc.preprocess_feat(data=data)
    tech_indicator_lst = featProc.techIndicatorLst 
    stock_num = data_dict['train']['stock'].nunique()
    print("Data has been processed..")
    
    return data_dict, tech_indicator_lst, stock_num

def setup_environments(config, data_dict, tech_indicator_lst, stock_num, mkt_observer=None):
    """
    Set up training, validation, and test environments
    
    Args:
        config: Configuration object
        data_dict: Dictionary containing processed data
        tech_indicator_lst: List of technical indicators
        stock_num: Number of stocks
        mkt_observer: Market observer object (optional)
    
    Returns:
        tuple: (env_train, env_valid, env_test)
    """
    # Initialize training environment
    trainInvest_env_para = config.invest_env_para 
    extra_train = data_dict.get('extra_train', None)
    
    env_train = StockPortfolioEnv(
        config=config, rawdata=data_dict['train'], mode='train', 
        stock_num=stock_num, action_dim=stock_num, 
        tech_indicator_lst=tech_indicator_lst, 
        extra_data=extra_train, 
        mkt_observer=mkt_observer, 
        **trainInvest_env_para
    )
    
    # Initialize validation environment if dates are specified
    env_valid = None
    if (config.valid_date_start is not None) and (config.valid_date_end is not None):
        validInvest_env_para = config.invest_env_para 
        extra_valid = data_dict.get('extra_valid', None)
        
        env_valid = StockPortfolioEnv(
            config=config, rawdata=data_dict['valid'], mode='valid', 
            stock_num=stock_num, action_dim=stock_num, 
            tech_indicator_lst=tech_indicator_lst,
            extra_data=extra_valid,
            mkt_observer=mkt_observer, 
            **validInvest_env_para
        )
    
    # Initialize test environment if dates are specified
    env_test = None
    if (config.test_date_start is not None) and (config.test_date_end is not None):
        testInvest_env_para = config.invest_env_para 
        extra_test = data_dict.get('extra_test', None)
        
        env_test = StockPortfolioEnv(
            config=config, rawdata=data_dict['test'], mode='test', 
            stock_num=stock_num, action_dim=stock_num, 
            tech_indicator_lst=tech_indicator_lst,
            extra_data=extra_test,
            mkt_observer=mkt_observer, 
            **testInvest_env_para
        )
    
    return env_train, env_valid, env_test

def create_market_observer(config, stock_num):
    """
    Create market observer if enabled
    
    Args:
        config: Configuration object
        stock_num: Number of stocks
    
    Returns:
        Market observer object or None
    """
    if not config.enable_market_observer:
        return None
        
    if ('ma' in config.mktobs_algo) or ('dc' in config.mktobs_algo):
        return MarketObserver_Algorithmic(config=config, action_dim=stock_num)
    else:
        return MarketObserver(config=config, action_dim=stock_num)

def train_model(config, env_train, env_valid, env_test):
    """
    Train the model and measure execution time
    
    Args:
        config: Configuration object
        env_train: Training environment
        env_valid: Validation environment
        env_test: Test environment
    """
    # Load RL model
    ModelCls = model_select(model_name=config.rl_model_name, mode=config.mode)
    model_para_dict = config.model_para
    po_model = ModelCls(env=env_train, **model_para_dict)
    
    # Setup training parameters
    total_timesteps = int(config.num_epochs * env_train.totalTradeDay)
    log_interval = 10
    callback = PoCallback(config=config, train_env=env_train, valid_env=env_valid, test_env=env_test)
    
    # Track execution time using multiple methods
    print('Training Start', flush=True)
    cpt_start = time.process_time()
    perft_start = time.perf_counter()
    timeit_default = timeit.default_timer()
    
    # Use timeit for more accurate timing
    my_globals = globals()
    my_globals.update({
        'po_model': po_model, 
        'total_timesteps': total_timesteps, 
        'callback': callback, 
        'log_interval': log_interval
    })
    
    t = timeit.Timer(
        stmt='po_model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=log_interval)', 
        globals=my_globals
    )
    
    time_usage = t.timeit(number=1)
    cpt_usage = time.process_time() - cpt_start
    perf_usage = time.perf_counter() - perft_start
    timeit_usage = timeit.default_timer() - timeit_default
    
    # Report timing results
    print(f"Time usage for {config.num_epochs} epochs: {np.round(time_usage, 2)}s, "
          f"cpu time: {np.round(cpt_usage, 2)}s, "
          f"perf_counter: {np.round(perf_usage, 2)}s, "
          f"timeit_default: {np.round(timeit_usage, 2)}s")
    print("-*" * 20)
    
    # Cleanup
    del po_model
    print("Training Done...", flush=True)

def run_rl_only(config):
    """
    Run the single-agent RL-based framework (TD3-Profit, TD3-PR, TD3-SR)
    
    Args:
        config: Configuration object
    """
    # Load and process data
    data_dict, tech_indicator_lst, stock_num = load_and_process_data(config)
    
    # Setup environments
    env_train, env_valid, env_test = setup_environments(
        config, data_dict, tech_indicator_lst, stock_num
    )
    
    # Train model
    train_model(config, env_train, env_valid, env_test)

def run_rl_controller(config):
    """
    Run the MASA framework
    
    Args:
        config: Configuration object
    """
    # Load and process data
    data_dict, tech_indicator_lst, stock_num = load_and_process_data(config)
    
    # Create market observer
    mkt_observer = create_market_observer(config, stock_num)
    
    # Validate that required data is available for MASA framework
    if (config.valid_date_start is not None and config.valid_date_end is not None) and \
       'extra_valid' not in data_dict:
        raise ValueError("No validation set is provided for training")
    
    if (config.test_date_start is not None and config.test_date_end is not None) and \
       'extra_test' not in data_dict:
        raise ValueError("No test set is provided for training")
    
    # Setup environments
    env_train, env_valid, env_test = setup_environments(
        config, data_dict, tech_indicator_lst, stock_num, mkt_observer
    )
    
    # Train model
    train_model(config, env_train, env_valid, env_test)