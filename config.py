#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
---------------------------------
 Name:         config.py
 Description:  configuration file that loads settings from YAML
 Author:       MASA (Refactored)

Referenced Repository in Github
Imple. of compared methods: https://github.com/ZhengyaoJiang/PGPortfolio/blob/48cc5a4af5edefd298e7801b95b0d4696f5175dd/pgportfolio/tdagent/tdagent.py#L7
RL-based agent (TD3 imple.): Baselines3 (https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
Trading environment: FinRL (https://github.com/AI4Finance-Foundation/FinRL)
Technical indicator imple.: TA-Lib (https://github.com/TA-Lib/ta-lib-python)
Second-order cone programming solver: CVXOPT (http://cvxopt.org/) 
---------------------------------
'''

import numpy as np
import os
import pandas as pd
import time
import datetime
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from RL_controller.TD3_controller import TD3PolicyOriginal

class Config:
    def __init__(self, config_path: str = 'config.yaml', seed_num: int = None, current_date: str = None):
        """
        Initialize configuration by loading from YAML file
        
        Args:
            config_path: Path to the YAML configuration file
            seed_num: Random seed (overrides config file if provided)
            current_date: Current date string (overrides default if provided)
        """
        # Load configuration from YAML file
        self.config_path = config_path
        self._load_config()
        
        # Override seed if provided
        if seed_num is not None:
            self.seed_num = seed_num
        
        # Process configuration
        self._process_benchmark_algorithm()
        self._setup_directories(current_date)
        self._setup_dates()
        self._setup_indicators()
        
        # Load additional configurations
        self.load_model_config()
        self.load_market_observer_config()
        self.load_para()
        
        # Validate configuration
        self._validate_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Load general settings
            self.notes = config.get('notes', 'AAMAS MASA Implementation')
            self.benchmark_algo = config.get('benchmark_algo', 'MASA-dc')
            self.market_name = config.get('market_name', 'DJIA')
            self.topK = config.get('topK', 5)
            self.num_epochs = config.get('num_epochs', 1)
            
            # Load directory settings
            self.dataDir = config.get('dataDir', './data')
            self.seed_num = config.get('seed_num', 2022)
            
            # Load trading parameters
            self.trade_pattern = config.get('trade_pattern', 1)
            self.lambda_1 = config.get('lambda_1', 1000.0)
            self.lambda_2 = config.get('lambda_2', 10.0)
            self.risk_default = config.get('risk_default', 0.017)
            self.risk_up_bound = config.get('risk_up_bound', 0.012)
            self.risk_hold_bound = config.get('risk_hold_bound', 0.014)
            self.risk_down_bound = config.get('risk_down_bound', 0.017)
            self.risk_market = config.get('risk_market', 0.001)
            self.cbf_gamma = config.get('cbf_gamma', 0.7)
            
            # Load period mode
            self.period_mode = config.get('period_mode', 1)
            self.date_split_dict = config.get('date_split', {})
            
            # Load financial parameters
            self.tradeDays_per_year = config.get('tradeDays_per_year', 252)
            self.tradeDays_per_month = config.get('tradeDays_per_month', 21)
            self.mkt_rf = config.get('mkt_rf', {'SP500': 1.6575, 'CSI300': 3.037, 'DJIA': 1.6575})
            self.market_close_time = config.get('market_close_time', {'CSI300': '15:00:00'})
            
            # Load investment environment parameters
            invest_env = config.get('invest_env', {})
            self.invest_env_para = {
                'max_shares': invest_env.get('max_shares', 100),
                'initial_asset': invest_env.get('initial_asset', 1000000),
                'transaction_cost': invest_env.get('transaction_cost', 0.0003),
                'slippage': invest_env.get('slippage', 0.001),
                'seed_num': self.seed_num,
                'reward_scaling': config.get('rl', {}).get('reward_scaling', 1),
                'norm_method': config.get('norm_method', 'sum')
            }
            
            # Load RL parameters
            rl_config = config.get('rl', {})
            self.reward_scaling = rl_config.get('reward_scaling', 1)
            self.learning_rate = rl_config.get('learning_rate', 0.0001)
            self.batch_size = rl_config.get('batch_size', 50)
            self.gradient_steps = rl_config.get('gradient_steps', 1)
            self.train_freq = rl_config.get('train_freq', [1, 'episode'])
            self.ars_trial = rl_config.get('ars_trial', 10)
            
            # Load algorithm lists
            self.only_long_algo_lst = config.get('only_long_algo_lst', ['CRP', 'EG', 'OLMAR', 'PAMR', 'RMR'])
            self.use_cash_algo_lst = config.get('use_cash_algo_lst', ['RAT', 'EIIE', 'PPN'])
            
            # Load price prediction model
            self.pricePredModel = config.get('pricePredModel', 'MA')
            self.cov_lookback = config.get('tech_indicator', {}).get('dailyReturn_lookback', 5)
            self.norm_method = config.get('norm_method', 'sum')
            
            # Market observer config
            market_observer = config.get('market_observer', {})
            self.dc_threshold = market_observer.get('dc', {}).get('threshold', [0.01])
            
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            print("Using default configuration values.")
            # Set defaults for critical values
            self.notes = 'AAMAS MASA Implementation'
            self.benchmark_algo = 'MASA-dc'
            self.market_name = 'DJIA'
            self.topK = 5
            self.num_epochs = 1
            self.dataDir = './data'
            self.seed_num = 2022 if seed_num is None else seed_num
            self.period_mode = 1
            self.pricePredModel = 'MA'
            self.cov_lookback = 5
            self.norm_method = 'sum'
            self.risk_default = 0.017
            self.risk_up_bound = 0.012
            self.risk_hold_bound = 0.014
            self.risk_down_bound = 0.017
            self.risk_market = 0.001
            self.cbf_gamma = 0.7
            self.reward_scaling = 1
            self.learning_rate = 0.0001
            self.batch_size = 50
            self.gradient_steps = 1
            self.train_freq = [1, 'episode']
            self.ars_trial = 10
            self.trade_pattern = 1
            self.lambda_1 = 1000.0
            self.lambda_2 = 10.0
            self.tradeDays_per_year = 252
            self.tradeDays_per_month = 21
            self.mkt_rf = {'SP500': 1.6575, 'CSI300': 3.037, 'DJIA': 1.6575}
            self.market_close_time = {'CSI300': '15:00:00'}
            self.invest_env_para = {
                'max_shares': 100,
                'initial_asset': 1000000,
                'reward_scaling': 1,
                'norm_method': 'sum',
                'transaction_cost': 0.0003,
                'slippage': 0.001,
                'seed_num': self.seed_num
            }
            self.only_long_algo_lst = ['CRP', 'EG', 'OLMAR', 'PAMR', 'RMR']
            self.use_cash_algo_lst = ['RAT', 'EIIE', 'PPN']
            self.dc_threshold = [0.01]
            
            # Default date split dictionary
            self.date_split_dict = {
                1: {
                    'train_date_start': '2019-01-10',
                    'train_date_end': '2023-12-01',
                    'valid_date_start': '2023-12-02',
                    'valid_date_end': '2024-12-02',
                    'test_date_start': '2024-12-03',
                    'test_date_end': '2025-05-03'
                }
            }
    
    def _process_benchmark_algorithm(self) -> None:
        """Process benchmark algorithm setting"""
        if 'TD3' in self.benchmark_algo:
            self.rl_model_name = 'TD3'
            self.mode = 'RLonly'
            self.mktobs_algo = None
            obj_name = self.benchmark_algo.split('-')[1]
            if obj_name == 'Profit':
                self.trained_best_model_type = 'max_capital'
            elif obj_name == 'PR':
                self.trained_best_model_type = 'pr_loss'
            elif obj_name == 'SR':
                self.trained_best_model_type = 'sr_loss'
            else:
                raise ValueError(f"Undefined obj_name [{obj_name}] of {self.benchmark_algo}.")
        
        elif 'MASA' in self.benchmark_algo:
            self.rl_model_name = 'TD3'
            self.mode = 'RLcontroller'
            self.mktobs_algo = f"{self.benchmark_algo.split('-')[1]}_1"
            self.trained_best_model_type = 'js_loss'
        
        else:
            # Baseline models
            self.rl_model_name = self.benchmark_algo
            self.mode = 'Benchmark'
            self.mktobs_algo = None
            self.trained_best_model_type = 'max_capital'
        
        # Set feature flags based on mode
        self.is_enable_dynamic_risk_bound = True
        self.enable_controller = True
        self.enable_market_observer = True
        
        if self.mode != 'RLcontroller':
            self.enable_controller = False
            self.enable_market_observer = False
            self.is_enable_dynamic_risk_bound = False
        
        # Adjust topK for DJIA if needed
        if (self.market_name == 'DJIA') and (self.topK == 30):
            self.topK = 29  # Only 29 stocks having complete data in the DJIA during that period
            
        # Configure DC feature generation
        if self.mktobs_algo is not None:
            self.is_gen_dc_feat = 'dc' in self.mktobs_algo
        else:
            self.is_gen_dc_feat = False
            
        # Set temporary name for logging and output
        self.tmp_name = f'Cls3_{self.mode}_{self.mktobs_algo}_{self.topK}_M{self.period_mode}_{self.market_name}_{self.trained_best_model_type}'
    
    def _setup_directories(self, current_date: Optional[str] = None) -> None:
        """Setup directories for results"""
        if current_date is None:
            self.cur_datetime = datetime.datetime.now().strftime('%Y-%m-%d')
        else:
            self.cur_datetime = current_date
            
        self.res_dir = os.path.join('./res', self.mode, self.rl_model_name, 
                                   f'{self.market_name}-{self.topK}', self.cur_datetime)
        os.makedirs(self.res_dir, exist_ok=True)
        
        self.res_model_dir = os.path.join(self.res_dir, 'model')
        os.makedirs(self.res_model_dir, exist_ok=True)
        
        self.res_img_dir = os.path.join(self.res_dir, 'graph')
        os.makedirs(self.res_img_dir, exist_ok=True)
    
    def _setup_dates(self) -> None:
        """Setup training, validation and testing dates"""
        if self.period_mode in self.date_split_dict:
            date_split = self.date_split_dict[self.period_mode]
        else:
            # Default date split as fallback
            date_split = {
                'train_date_start': '2019-01-10',
                'train_date_end': '2023-12-01',
                'valid_date_start': '2023-12-02',
                'valid_date_end': '2024-12-02',
                'test_date_start': '2024-12-03',
                'test_date_end': '2025-05-03'
            }
            
        self.train_date_start = pd.Timestamp(date_split['train_date_start'])
        self.train_date_end = pd.Timestamp(date_split['train_date_end'])
        
        if date_split.get('valid_date_start') and date_split.get('valid_date_end'):
            self.valid_date_start = pd.Timestamp(date_split['valid_date_start'])
            self.valid_date_end = pd.Timestamp(date_split['valid_date_end'])
        else:
            self.valid_date_start = None
            self.valid_date_end = None
            
        if date_split.get('test_date_start') and date_split.get('test_date_end'):
            self.test_date_start = pd.Timestamp(date_split['test_date_start'])
            self.test_date_end = pd.Timestamp(date_split['test_date_end'])
        else:
            self.test_date_start = None
            self.test_date_end = None
    
    def _setup_indicators(self) -> None:
        """Setup technical indicators"""
        # Extract from YAML if available, otherwise use defaults
        self.tech_indicator_talib_lst = []
        self.tech_indicator_extra_lst = ['CHANGE']
        self.tech_indicator_input_lst = self.tech_indicator_talib_lst + self.tech_indicator_extra_lst
        
        self.dailyRetun_lookback = self.cov_lookback
        self.otherRef_indicator_ma_window = 5
        self.enable_cov_features = False
        
        self.otherRef_indicator_lst = [
            f'MA-{self.otherRef_indicator_ma_window}', 
            f'DAILYRETURNS-{self.dailyRetun_lookback}'
        ]
    
    def load_model_config(self) -> None:
        """Load model configuration"""
        # These could come from YAML in a more complete implementation
        self.use_features = ['close', 'open', 'high', 'low']
        self.window_size = 31
        self.po_lr = 0.0001
        self.po_weight_decay = 0.001
    
    def load_market_observer_config(self) -> None:
        """Load market observer configuration"""
        self.freq = '1d'
        self.finefreq = '60m'
        self.fine_window_size = 4
        self.feat_scaler = 10
        
        self.hidden_vec_loss_weight = 1e4
        self.sigma_loss_weight = 1e5
        self.lambda_min = 0.0
        self.lambda_max = 1.0
        self.sigma_min = 0.0
        self.sigma_max = 1.0
        
        self.finestock_feat_cols_lst = []
        self.finemkt_feat_cols_lst = []
        for ifeat in self.use_features:
            for iwin in range(1, self.fine_window_size+1):
                self.finestock_feat_cols_lst.append(f'stock_{self.finefreq}_{ifeat}_w{iwin}')
                self.finemkt_feat_cols_lst.append(f'mkt_{self.finefreq}_{ifeat}_w{iwin}')
    
    def load_para(self) -> None:
        """Load model parameters"""
        # Determine policy name based on configuration
        if self.enable_market_observer:
            if self.rl_model_name == 'TD3':
                policy_name = 'TD3PolicyAdj'
            else:
                raise ValueError(f"Cannot specify the {self.rl_model_name} policy name when enabling market observer.")
        else:
            if self.rl_model_name == 'TD3':
                policy_name = TD3PolicyOriginal
            else:
                if self.mode in ['RLonly', 'RLcontroller']:
                    raise ValueError(f"Cannot specify the {self.rl_model_name} policy name when using stable-baseline.")
                else:
                    policy_name = "MlpPolicy"
        
        # Base parameters for all models
        base_para = {
            'policy': policy_name,
            'learning_rate': self.learning_rate,
            'buffer_size': 1000000,
            'learning_starts': 100,
            'batch_size': self.batch_size,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': (self.train_freq[0], self.train_freq[1]),
            'gradient_steps': self.gradient_steps,
            'action_noise': None,
            'replay_buffer_class': None,
            'replay_buffer_kwargs': None,
            'optimize_memory_usage': False,
            'tensorboard_log': None,
            'policy_kwargs': None,
            'verbose': 1,
            'seed': self.seed_num,
            'device': 'auto',
            '_init_setup_model': True,
        }
        
        # Algorithm-specific parameters
        algo_para = {
            'TD3': {
                'policy_delay': 2,
                'target_policy_noise': 0.2,
                'target_noise_clip': 0.5,
            },
            'SAC': {
                'ent_coef': 'auto',
                'target_update_interval': 1,
                'target_entropy': 'auto',
                'use_sde': False,
                'sde_sample_freq': -1,
                'use_sde_at_warmup': False,
            },
            'PPO': {
                'n_steps': 100
            },
        }
        
        # Adjust network architecture for larger portfolios
        if (self.topK == 20) or (self.topK == 30):
            algo_para['TD3']['policy_kwargs'] = {
                'net_arch': [1024, 512, 128],
            }
        
        # Parameters to remove for specific algorithms
        algo_para_rm_from_base = {
            'PPO': [
                'buffer_size', 'learning_starts', 'tau', 'train_freq',
                'gradient_steps', 'action_noise', 'replay_buffer_class',
                'replay_buffer_kwargs', 'optimize_memory_usage'
            ]
        }
        
        # Combine parameters
        if self.rl_model_name in algo_para:
            self.model_para = {**base_para, **algo_para[self.rl_model_name]}
        else:
            self.model_para = base_para
            
        # Remove unnecessary parameters for specific algorithms
        if self.rl_model_name in algo_para_rm_from_base:
            for rm_field in algo_para_rm_from_base[self.rl_model_name]:
                if rm_field in self.model_para:
                    del self.model_para[rm_field]
    
    def _validate_config(self) -> None:
        """Validate configuration for correctness"""
        if self.risk_default <= self.risk_market:
            raise ValueError(f"The boundary of safe risk[{self.risk_default}] should not be less than/equal to the market risk[{self.risk_market}].")
            
        if self.mode == 'Benchmark':
            self.trained_best_model_type = 'max_capital'
            
        if self.mode == 'RLonly':
            if self.trained_best_model_type not in ['max_capital', 'pr_loss', 'sr_loss']:
                raise ValueError(f"The trained_best_model_type[{self.trained_best_model_type}] of {self.mode} should be in ['max_capital', 'pr_loss', 'sr_loss']")
    
    def print_config(self) -> None:
        """Print configuration details for logging"""
        log_str = '=' * 30 + '\n'
        para_str = f'{self.notes} \n'
        log_str = log_str + para_str
        para_str = f'mode: {self.mode}, rl_model_name: {self.rl_model_name}, market_name: {self.market_name}, topK: {self.topK}, dataDir: {self.dataDir}, enable_controller: {self.enable_controller}, \n'
        log_str = log_str + para_str
        para_str = f'trade_pattern: {self.trade_pattern} \n'
        log_str = log_str + para_str
        para_str = f'period_mode: {self.period_mode}, num_epochs: {self.num_epochs}, cov_lookback: {self.cov_lookback}, norm_method: {self.norm_method}, benchmark_algo: {self.benchmark_algo}, trained_best_model_type: {self.trained_best_model_type}, pricePredModel: {self.pricePredModel}, \n'
        log_str = log_str + para_str
        para_str = f'is_enable_dynamic_risk_bound: {self.is_enable_dynamic_risk_bound}, risk_market: {self.risk_market}, risk_default: {self.risk_default}, cbf_gamma: {self.cbf_gamma}, ars_trial: {self.ars_trial} \n'
        log_str = log_str + para_str
        para_str = f'cur_datetime: {self.cur_datetime}, res_dir: {self.res_dir}, tradeDays_per_year: {self.tradeDays_per_year}, tradeDays_per_month: {self.tradeDays_per_month}, seed_num: {self.seed_num}, \n'
        log_str = log_str + para_str
        para_str = f'train_date_start: {self.train_date_start}, train_date_end: {self.train_date_end}, valid_date_start: {self.valid_date_start}, valid_date_end: {self.valid_date_end}, test_date_start: {self.test_date_start}, test_date_end: {self.test_date_end}, \n'
        log_str = log_str + para_str
        para_str = f'tech_indicator_input_lst: {self.tech_indicator_input_lst}, \n'
        log_str = log_str + para_str
        para_str = f'otherRef_indicator_lst: {self.otherRef_indicator_lst}, enable_cov_features: {self.enable_cov_features} \n'
        log_str = log_str + para_str
        para_str = f'tmp_name: {self.tmp_name}, mkt_rf: {self.mkt_rf} \n'
        log_str = log_str + para_str
        para_str = f'invest_env_para: {self.invest_env_para}, \n'
        log_str = log_str + para_str
        para_str = f'model_para: {self.model_para}, \n'
        log_str = log_str + para_str
        para_str = f'only_long_algo_lst: {self.only_long_algo_lst}, \n'
        log_str = log_str + para_str
        para_str = f'lstr para: use_features: {self.use_features}, window_size: {self.window_size}, freq: {self.freq}, finefreq: {self.finefreq}, fine_window_size: {self.fine_window_size}, \n'
        log_str = log_str + para_str
        para_str = f'enable_market_observer: {self.enable_market_observer}, mktobs_algo: {self.mktobs_algo}, feat_scaler: {self.feat_scaler} \n'
        log_str = log_str + para_str
        log_str = log_str + '=' * 30 + '\n'

        print(log_str, flush=True)
        
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to a YAML file
        
        Args:
            output_path: Path to save the configuration file (defaults to config_path)
        """
        if output_path is None:
            output_path = self.config_path
            
        # Convert current configuration to a dictionary
        config_dict = {
            'notes': self.notes,
            'benchmark_algo': self.benchmark_algo,
            'market_name': self.market_name,
            'topK': self.topK,
            'num_epochs': self.num_epochs,
            'dataDir': self.dataDir,
            'seed_num': self.seed_num,
            'trade_pattern': self.trade_pattern,
            'lambda_1': self.lambda_1,
            'lambda_2': self.lambda_2,
            'risk_default': self.risk_default,
            'risk_up_bound': self.risk_up_bound,
            'risk_hold_bound': self.risk_hold_bound,
            'risk_down_bound': self.risk_down_bound,
            'risk_market': self.risk_market,
            'cbf_gamma': self.cbf_gamma,
            'period_mode': self.period_mode,
            'date_split': self.date_split_dict,
            'pricePredModel': self.pricePredModel,
            'cov_lookback': self.cov_lookback,
            'norm_method': self.norm_method,
            'tradeDays_per_year': self.tradeDays_per_year,
            'tradeDays_per_month': self.tradeDays_per_month,
            'mkt_rf': self.mkt_rf,
            'market_close_time': self.market_close_time,
            'invest_env': {
                'max_shares': self.invest_env_para['max_shares'],
                'initial_asset': self.invest_env_para['initial_asset'],
                'transaction_cost': self.invest_env_para['transaction_cost'],
                'slippage': self.invest_env_para['slippage']
            },
            'rl': {
                'reward_scaling': self.reward_scaling,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'gradient_steps': self.gradient_steps,
                'train_freq': self.train_freq,
                'ars_trial': self.ars_trial
            },
            'model': {
                'window_size': self.window_size,
                'po_lr': self.po_lr,
                'po_weight_decay': self.po_weight_decay,
                'use_features': self.use_features
            },
            'market_observer': {
                'freq': self.freq,
                'finefreq': self.finefreq,
                'fine_window_size': self.fine_window_size,
                'feat_scaler': self.feat_scaler,
                'hidden_vec_loss_weight': self.hidden_vec_loss_weight,
                'sigma_loss_weight': self.sigma_loss_weight,
                'lambda_min': self.lambda_min,
                'lambda_max': self.lambda_max,
                'sigma_min': self.sigma_min,
                'sigma_max': self.sigma_max,
                'dc': {
                    'threshold': self.dc_threshold
                }
            },
            'only_long_algo_lst': self.only_long_algo_lst,
            'use_cash_algo_lst': self.use_cash_algo_lst
        }
        
        # Write to file
        with open(output_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)