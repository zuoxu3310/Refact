#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name: tradeEnv.py  
 Description: Define the trading environment for the trading agent.
 Author: MASA
--------------------------------
'''
# Import all components from refactored modules to maintain backward compatibility
from utils.trading.environments import StockPortfolioEnv, StockPortfolioEnv_cash
from utils.trading.normalization import softmax_normalization, sum_normalization
from utils.trading.metrics import calculate_portfolio_metrics
from utils.trading.visualization import save_action_memory, save_profile
from utils.trading.market_observer import run_market_observer
from utils.trading.reward_functions import calculate_reward

# Export all components that were accessible in the original file
__all__ = [
    'StockPortfolioEnv',
    'StockPortfolioEnv_cash',
    'softmax_normalization',
    'sum_normalization',
    'calculate_portfolio_metrics',
    'save_action_memory',
    'save_profile',
    'run_market_observer',
    'calculate_reward'
]