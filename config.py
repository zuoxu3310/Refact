#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
import time
import datetime
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from RL_controller.TD3_controller import TD3PolicyOriginal

from config_helpers.yaml_loader import load_yaml_config
from config_helpers.setup_routines import setup_directories, setup_dates, setup_indicators
from config_helpers.param_loader import load_model_config, load_market_observer_config, load_parameters
from config_helpers.config_io import generate_config_summary_string, save_config_to_yaml


class Config:
    def __init__(self, config_path: str = 'config.yaml', seed_num: int = None, current_date: str = None):
        """
        Initialize configuration by loading from YAML file
        
        Args:
            config_path: Path to the YAML configuration file
            seed_num: Random seed (overrides config file if provided)
            current_date: Current date string (overrides default if provided)
        """
        # Load configuration from YAML file using helper
        load_yaml_config(self, config_path, seed_num)
        
        # Process configuration
        self._process_benchmark_algorithm()
        
        # Setup using helper functions
        setup_directories(self, current_date)
        setup_dates(self)
        setup_indicators(self)
        
        # Load additional configurations using helper functions
        load_model_config(self)
        load_market_observer_config(self)
        load_parameters(self) # Renamed from load_para for clarity
        
        # Validate configuration
        self._validate_config()
    
    def _load_config(self) -> None:
        """
        This method is now handled by load_yaml_config in config_helpers.yaml_loader.
        Keeping this as a stub or removing it depends on whether any internal calls might still expect it.
        For now, let's assume it's fully replaced.
        """
        pass

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
        log_str = generate_config_summary_string(self)
        print(log_str, flush=True)
        
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to a YAML file
        
        Args:
            output_path: Path to save the configuration file (defaults to config_path)
        """
        save_config_to_yaml(self, output_path)