#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         environments.py
 Description:  Trading environments classes.
 Author:       MASA
---------------------------------
'''
import numpy as np
import os
import pandas as pd
import time
import copy
from gym.utils import seeding
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
import scipy.stats as spstats

from .normalization import get_weights_normalization_fn
from .metrics import calculate_portfolio_metrics
from .visualization import save_action_memory, save_profile
from .market_observer import run_market_observer
from .reward_functions import calculate_reward

class StockPortfolioEnv(gym.Env):
    """
    Stock portfolio trading environment.
    
    This environment simulates stock trading with portfolio allocation.
    """
    def __init__(self, config, rawdata, mode, stock_num, action_dim, tech_indicator_lst, max_shares,
                 initial_asset=1000000, reward_scaling=1, norm_method='sum', transaction_cost=0.001, slippage=0.001, 
                 seed_num=2022, extra_data=None, mkt_observer=None):
        """
        Initialize the stock portfolio environment.
        
        Args:
            config: Configuration object with parameters
            rawdata: Raw stock price data
            mode: Mode of operation ('train', 'valid', 'test')
            stock_num: Number of stocks in the portfolio
            action_dim: Dimension of the action space
            tech_indicator_lst: List of technical indicators
            max_shares: Maximum number of shares
            initial_asset: Initial portfolio value
            reward_scaling: Scaling factor for reward
            norm_method: Method for normalizing weights ('sum' or 'softmax')
            transaction_cost: Cost of transactions as percentage
            slippage: Price slippage as percentage
            seed_num: Random seed
            extra_data: Extra data for environment
            mkt_observer: Market observer instance
        """
        self.config = config
        self.rawdata = rawdata
        self.mode = mode  # train, valid, test
        self.stock_num = stock_num  # Number of stocks
        self.action_dim = action_dim  # Number of assets
        self.tech_indicator_lst = tech_indicator_lst
        self.tech_indicator_lst_wocov = copy.deepcopy(self.tech_indicator_lst)  # without cov feature
        if 'cov' in self.tech_indicator_lst_wocov:
            self.tech_indicator_lst_wocov.remove('cov')
        self.max_shares = max_shares  # Maximum number of shares
        self.seed_num = seed_num 
        self.seed(seed=self.seed_num)
        self.epoch = 0
        self.curTradeDay = 0
        self.eps = 1e-6

        self.initial_asset = initial_asset  # Initial portfolio value
        self.reward_scaling = reward_scaling 
        self.norm_method = norm_method
        self.transaction_cost = transaction_cost  # 0.001
        self.slippage = slippage  # 0.001 for one-side, 0.002 for two-side
        self.cur_slippage_drift = np.random.random(self.stock_num) * (self.slippage * 2) - self.slippage
        
        # Extra data for market observer
        if extra_data is not None:
            self.extra_data = extra_data
        else:
            self.extra_data = None

        # Set weight normalization function
        self.weights_normalization = get_weights_normalization_fn(norm_method)
        
        # State space dimension
        if self.config.enable_cov_features:        
            self.state_dim = ((len(self.tech_indicator_lst_wocov)+self.stock_num) * self.stock_num) + 1  # +1: current portfolio value 
        else:
            self.state_dim = (len(self.tech_indicator_lst_wocov) * self.stock_num) + 1  # +1: current portfolio value
        
        # Add market observer dimension if enabled
        if self.config.enable_market_observer:
            self.state_dim = self.state_dim + self.stock_num
            self.mkt_observer = mkt_observer
        else:
            self.mkt_observer = None
        
        # Set action space
        if self.config.benchmark_algo in self.config.only_long_algo_lst:
            # Long only
            self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim, ))
            self.bound_flag = 1  # 1 for long and long+short, -1 for short
        else:
            if self.config.trade_pattern == 1:
                # Long only
                self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim, ))
                self.bound_flag = 1  # 1 for long and long+short, -1 for short
            else:
                raise ValueError("Unexpected trade pattern: {}".format(self.config.trade_pattern))
        
        # Set observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim, ))

        # Prepare raw data
        self.rawdata.sort_values(['date', 'stock'], ascending=True, inplace=True)
        self.rawdata.index = self.rawdata.date.factorize()[0]
        self.totalTradeDay = len(self.rawdata['date'].unique())
        self.stock_lst = np.sort(self.rawdata['stock'].unique())

        # Get initial data
        self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
        self.curData.sort_values(['stock'], ascending=True, inplace=True)
        self.curData.reset_index(drop=True, inplace=True)

        # Initialize state
        if self.config.enable_cov_features:   
            self.covs = np.array(self.curData['cov'].values[0])
            self.state = np.append(self.covs, np.transpose(self.curData[self.tech_indicator_lst_wocov].values), axis=0)
        else:
            self.state = np.transpose(self.curData[self.tech_indicator_lst_wocov].values)
        self.state = self.state.flatten()
        self.state = np.append(self.state, [0], axis=0)
        self.ctl_state = {k:np.array(list(self.curData[k].values)) for k in self.config.otherRef_indicator_lst}
        self.terminal = False

        # Initialize history lists
        self.profit_lst = [0]  # percentage of portfolio daily returns
        cur_risk_boundary, stock_ma_price = run_market_observer(self, stage='init')  # after curData and state, before cur_risk_boundary
        if stock_ma_price is not None:
            self.ctl_state['MA-{}'.format(self.config.otherRef_indicator_ma_window)] = stock_ma_price
        self.cur_capital = self.initial_asset
        
        self.cvar_lst = [0]
        self.cvar_raw_lst = [0]

        self.asset_lst = [self.initial_asset] 
        self.date_memory = [self.curData['date'].unique()[0]]
        self.reward_lst = [0]
        self.action_cbf_memeory = [np.array([0] * self.stock_num)]

        self.actions_memory = [np.array([1/self.stock_num]*self.stock_num) * self.bound_flag] 
        self.action_rl_memory = [np.array([1/self.stock_num]*self.stock_num) * self.bound_flag]

        self.risk_adj_lst = [cur_risk_boundary]
        self.is_last_ctrl_solvable = False
        self.risk_raw_lst = [0]  # For performance analysis. Record the risk without using risk controller during the validation/test period.
        self.risk_cbf_lst = [0]
        self.return_raw_lst = [self.initial_asset] 
        self.solver_stat = {'solvable': 0, 'insolvable': 0, 'stochastic_solvable': 0, 'stochastic_time': [], 'socp_solvable': 0, 'socp_time': []} 

        self.ctrl_weight_lst = [1.0]
        self.solvable_flag = []
        self.risk_pred_lst = []

        self.rl_reward_risk_lst = []
        self.rl_reward_profit_lst = []
        self.cnt1 = 0
        self.cnt2 = 0
        self.stepcount = 0

        # Timing variables
        self.start_cputime = time.process_time()
        self.start_systime = time.perf_counter()
        if self.mode == 'train':
            self.exclusive_cputime = 0
            self.exclusive_systime = 0
            
        # For saving profile
        self.profile_hist_field_lst = [            
            'ep', 'trading_days', 'annualReturn_pct', 'mdd', 'sharpeRatio', 'final_capital', 'volatility', 
            'calmarRatio', 'sterlingRatio',
            'netProfit', 'netProfit_pct', 'winRate',
            'vol_max', 'vol_min', 'vol_avg', 
            'risk_max', 'risk_min', 'risk_avg', 'riskRaw_max', 'riskRaw_min', 'riskRaw_avg',
            'dailySR_max', 'dailySR_min', 'dailySR_avg', 'dailySR_wocbf_max', 'dailySR_wocbf_min', 'dailySR_wocbf_avg',
            'dailyReturn_pct_max', 'dailyReturn_pct_min', 'dailyReturn_pct_avg',
            'sigReturn_max', 'sigReturn_min', 'mdd_high', 'mdd_low', 'mdd_high_date', 'mdd_low_date', 'sharpeRatio_wocbf',
            'reward_sum', 'final_capital_wocbf', 'cbf_contribution',
            'risk_downsideAtVol', 'risk_downsideAtVol_daily_max', 'risk_downsideAtVol_daily_min', 'risk_downsideAtVol_daily_avg',
            'risk_downsideAtValue_daily_max', 'risk_downsideAtValue_daily_min', 'risk_downsideAtValue_daily_avg',
            'cvar_max', 'cvar_min', 'cvar_avg', 'cvar_raw_max', 'cvar_raw_min', 'cvar_raw_avg',
            'solver_solvable', 'solver_insolvable', 'cputime', 'systime', 
        ]
        self.profile_hist_ep = {k: [] for k in self.profile_hist_field_lst}
        
        # Initialize market observer price
        self.mkt_last_close_price = 0

    def step(self, actions):
        """
        Take a step in the environment.
        
        Args:
            actions: Actions to take (portfolio weights)
            
        Returns:
            Tuple of (state, reward, done, info)
        """
        # Check if episode is over
        self.terminal = self.curTradeDay >= (self.totalTradeDay - 1)
        
        if self.terminal:
            # Apply transaction cost to final capital
            self.cur_capital = self.cur_capital * (1 - self.transaction_cost)
            self.asset_lst[-1] = self.cur_capital
            self.profit_lst[-1] = (self.cur_capital - self.asset_lst[-2]) / self.asset_lst[-2]   
            
            # Apply transaction cost to raw return without CBF
            if len(self.action_rl_memory) > 1:
                self.return_raw_lst[-1] = self.return_raw_lst[-1] * (1 - self.transaction_cost)

            # Train market observer at end of episode
            if (self.config.enable_market_observer) and (self.mode == 'train'):
                ori_profit_rate = np.append([1], np.array(self.return_raw_lst)[1:] / np.array(self.return_raw_lst)[:-1], axis=0)
                adj_profit_rate = np.array(self.profit_lst) + 1
                label_kwargs = {'mode': self.mode, 'ori_profit': ori_profit_rate, 'adj_profit': adj_profit_rate, 'ori_risk': np.array(self.risk_raw_lst), 'adj_risk': np.array(self.risk_cbf_lst)}
                self.mkt_observer.train(**label_kwargs)       
            
            # Calculate elapsed time
            self.end_cputime = time.process_time()
            self.end_systime = time.perf_counter()
            self.model_save_flag = True
            
            # Calculate investment metrics
            invest_profile = self.get_results()
            save_profile(self, invest_profile)

            return self.state, self.reward, self.terminal, {}
        
        else:
            # Normalize actions to get portfolio weights
            actions = np.reshape(actions, (-1))  # [1, num_of_stocks] or [num_of_stocks, ]
            weights = self.weights_normalization(actions)  # Unnormalized weights -> normalized weights 
            self.actions_memory.append(weights)
            
            # Apply transaction costs
            if self.curTradeDay == 0:
                self.cur_capital = self.cur_capital * (1 - self.transaction_cost)
            else:
                # Calculate price changes with slippage
                cur_p = np.array(self.curData['close'].values) * (1 + self.cur_slippage_drift)
                last_p = np.array(self.lastDayData['close'].values) * (1 + self.last_slippage_drift)
                x_p = cur_p / last_p
                last_action = np.array(self.actions_memory[-2])
                
                # Adjust for extreme price changes
                x_p_adj = np.where((x_p>=2)&(last_action<0), 2, x_p)
                sgn = np.sign(last_action)
                
                # Calculate adjusted weights
                adj_w_ay = sgn * (last_action * (x_p_adj - 1) + np.abs(last_action))
                adj_cap = np.sum((x_p_adj - 1) * last_action) + 1
                
                # Check for capital loss
                if (adj_cap <= 0) or np.all(adj_w_ay==0):
                    raise ValueError("Loss the whole capital! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(
                        self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay))
                
                # Normalize adjusted weights
                last_w_adj = adj_w_ay / adj_cap
                
                # Apply transaction costs
                self.cur_capital = self.cur_capital * (1 - (np.sum(np.abs(self.actions_memory[-1] - last_w_adj)) * self.transaction_cost))
                self.asset_lst[-1] = self.cur_capital
                self.profit_lst[-1] = (self.cur_capital - self.asset_lst[-2]) / self.asset_lst[-2]
                
                # Calculate returns without CBF
                if len(self.action_rl_memory) > 1:
                    last_rl_action = np.array(self.action_rl_memory[-2])
                    sgn_rl = np.sign(last_rl_action)
                    prev_rl_cap = self.return_raw_lst[-1]
                    
                    # Adjust for extreme price changes in RL actions
                    x_p_adjrl = np.where((x_p>=2)&(last_rl_action<0), 2, x_p)
                    adj_w_ay = sgn_rl * (last_rl_action * (x_p_adjrl - 1) + np.abs(last_rl_action))
                    adj_cap = np.sum((x_p_adjrl - 1) * last_rl_action) + 1
                    
                    # Check for capital loss with RL actions
                    if (adj_cap <= 0) or np.all(adj_w_ay==0):
                        print("Loss the whole capital if using RL actions only! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(
                            self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay))
                        adj_w_ay = np.array([1/self.stock_num]*self.stock_num) * self.bound_flag
                        adj_cap = 1
                    
                    # Normalize adjusted weights for RL
                    last_rlw_adj = adj_w_ay / adj_cap 
                    
                    # Apply transaction costs to raw return
                    return_raw = prev_rl_cap * (1 - (np.sum(np.abs(self.action_rl_memory[-1] - last_rlw_adj)) * self.transaction_cost))
                    self.return_raw_lst[-1] = return_raw

            # Advance to next day
            self.curTradeDay = self.curTradeDay + 1
            self.lastDayData = self.curData
            self.last_slippage_drift = self.cur_slippage_drift
            
            # Load new day's data
            self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
            self.curData.sort_values(['stock'], ascending=True, inplace=True)
            self.curData.reset_index(drop=True, inplace=True)
            
            # Update state
            if self.config.enable_cov_features:   
                self.covs = np.array(self.curData['cov'].values[0])
                self.state = np.append(self.covs, np.transpose(self.curData[self.tech_indicator_lst_wocov].values), axis=0)
            else:
                self.state = np.transpose(self.curData[self.tech_indicator_lst_wocov].values)
            self.state = self.state.flatten()
            
            # Update controller state
            self.ctl_state = {k:np.array(list(self.curData[k].values)) for k in self.config.otherRef_indicator_lst}
            
            # Record date
            cur_date = self.curData['date'].unique()[0]
            self.date_memory.append(cur_date)

            # Generate new slippage drift
            self.cur_slippage_drift = np.random.random(self.stock_num) * (self.slippage * 2) - self.slippage
            
            # Calculate price changes with slippage
            curDay_ClosePrice_withSlippage = np.array(self.curData['close'].values) * (1 + self.cur_slippage_drift)
            lastDay_ClosePrice_withSlippage = np.array(self.lastDayData['close'].values) * (1 + self.last_slippage_drift)
            rate_of_price_change = curDay_ClosePrice_withSlippage / lastDay_ClosePrice_withSlippage
            
            # Adjust for extreme price changes
            rate_of_price_change_adj = np.where((rate_of_price_change>=2)&(weights<0), 2, rate_of_price_change)
            
            # Calculate single day returns
            sigDayReturn = (rate_of_price_change_adj - 1) * weights  # [s1_pct, s2_pct, .., px_pct_returns]
            poDayReturn = np.sum(sigDayReturn)
            
            # Check for capital loss
            if poDayReturn <= (-1):
                raise ValueError("Loss the whole capital! [Day: {}, date: {}, poDayReturn: {}]".format(
                    self.curTradeDay, self.date_memory[-1], poDayReturn))

            # Update portfolio value
            updatePoValue = self.cur_capital * (poDayReturn + 1) 
            poDayReturn_withcost = (updatePoValue - self.cur_capital) / self.cur_capital  # Include transaction cost

            self.cur_capital = updatePoValue
            self.state = np.append(self.state, [np.log(self.cur_capital/self.initial_asset)], axis=0)  # Add log return to state
            
            # Record daily returns and portfolio value
            self.profit_lst.append(poDayReturn_withcost)
            self.asset_lst.append(self.cur_capital)

            # Get market observer info
            cur_risk_boundary, stock_ma_price = run_market_observer(
                self, stage='run', rate_of_price_change=np.array([rate_of_price_change])
            )
            
            # Update MA prices if available
            if stock_ma_price is not None:
                self.ctl_state['MA-{}'.format(self.config.otherRef_indicator_ma_window)] = stock_ma_price
                
            # Record risk adjustments
            self.risk_adj_lst.append(cur_risk_boundary)
            self.ctrl_weight_lst.append(1.0)       

            # Calculate risks
            daily_return_ay = np.array(list(self.curData['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)].values))
            cur_cov = np.cov(daily_return_ay) 
            self.risk_cbf_lst.append(np.sqrt(np.matmul(np.matmul(weights, cur_cov), weights.T)))  # Daily risk
            
            # Calculate risk without CBF
            w_rl = self.action_rl_memory[-1]
            w_rl = w_rl / np.sum(np.abs(w_rl))
            self.risk_raw_lst.append(np.sqrt(np.matmul(np.matmul(w_rl, cur_cov), w_rl.T)))

            # Calculate returns without CBF
            if self.curTradeDay == 1:
                prev_rl_cap = self.return_raw_lst[-1] * (1 - self.transaction_cost)
            else:
                prev_rl_cap = self.return_raw_lst[-1]
            
            # Adjust for extreme price changes in RL actions
            rate_of_price_change_adj_rawrl = np.where((rate_of_price_change>=2)&(w_rl<0), 2, rate_of_price_change)
            po_r_rl = np.sum((rate_of_price_change_adj_rawrl - 1) * w_rl)
            
            # Check for capital loss without CBF
            if po_r_rl <= (-1):
                raise ValueError("Loss the whole capital if using RL actions only! [Day: {}, date: {}, po_r_rl: {}]".format(
                    self.curTradeDay, self.date_memory[-1], po_r_rl))
                
            # Calculate return without CBF
            return_raw = prev_rl_cap * (po_r_rl + 1) 
            self.return_raw_lst.append(return_raw)

            # Calculate CVaR (Conditional Value at Risk)
            expected_r_series = daily_return_ay[:, -21:]
            expected_r_prev = np.mean(expected_r_series[:, -1:], axis=1)
            expected_r_prev = np.where((expected_r_prev>=1)&(weights<0), 1, expected_r_prev)
            expected_r = np.sum(np.reshape(expected_r_prev, (1, -1)) @ np.reshape(weights, (-1, 1)))
            expected_cov = np.cov(expected_r_series)
            expected_std = np.sum(np.sqrt(np.reshape(weights, (1, -1)) @ expected_cov @ np.reshape(weights, (-1, 1))))
            cvar_lz = spstats.norm.ppf(1-0.05)  # positive 1.65 for 95%(=1-alpha) confidence level
            cvar_Z = np.exp(-0.5*np.power(cvar_lz, 2)) / 0.05 / np.sqrt(2*np.pi)
            cvar_expected = -expected_r + expected_std * cvar_Z
            self.cvar_lst.append(cvar_expected)

            # Calculate CVaR without risk controller
            expected_r_prevrl = np.mean(expected_r_series[:, -1:], axis=1)
            expected_r_prevrl = np.where((expected_r_prevrl>=1)&(w_rl<0), 1, expected_r_prevrl)
            expected_r_raw = np.sum(np.reshape(expected_r_prevrl, (1, -1)) @ np.reshape(w_rl, (-1, 1)))
            expected_std_raw = np.sum(np.sqrt(np.reshape(w_rl, (1, -1)) @ expected_cov @ np.reshape(w_rl, (-1, 1))))
            cvar_expected_raw = -expected_r_raw + expected_std_raw * cvar_Z
            self.cvar_raw_lst.append(cvar_expected_raw)

            # Calculate reward
            self.reward, profit_part, risk_part, scaled_profit_part, scaled_risk_part = calculate_reward(
                self, weights, w_rl, poDayReturn_withcost
            )

            # Record reward components
            self.rl_reward_risk_lst.append(scaled_risk_part)
            self.rl_reward_profit_lst.append(scaled_profit_part)
            self.reward_lst.append(self.reward)
            self.model_save_flag = False
            
            return self.state, self.reward, self.terminal, {}

    def reset(self):
        """
        Reset the environment.
        
        Returns:
            Initial state
        """
        self.epoch = self.epoch + 1
        self.curTradeDay = 0

        # Load initial data
        self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
        self.curData.sort_values(['stock'], ascending=True, inplace=True)
        self.curData.reset_index(drop=True, inplace=True)
        
        # Initialize state
        if self.config.enable_cov_features:   
            self.covs = np.array(self.curData['cov'].values[0])
            self.state = np.append(self.covs, np.transpose(self.curData[self.tech_indicator_lst_wocov].values), axis=0)
        else:
            self.state = np.transpose(self.curData[self.tech_indicator_lst_wocov].values)
        self.state = self.state.flatten()
        self.state = np.append(self.state, [0], axis=0)
        self.ctl_state = {k:np.array(list(self.curData[k].values)) for k in self.config.otherRef_indicator_lst} 
        self.terminal = False

        # Reset history lists
        self.profit_lst = [0] 
        cur_risk_boundary, stock_ma_price = run_market_observer(self, stage='reset')  
        if stock_ma_price is not None:
            self.ctl_state['MA-{}'.format(self.config.otherRef_indicator_ma_window)] = stock_ma_price

        self.cur_capital = self.initial_asset
        self.cvar_lst = [0]
        self.cvar_raw_lst = [0]
        self.asset_lst = [self.initial_asset] 
        self.actions_memory = [np.array([1/self.stock_num]*self.stock_num) * self.bound_flag]
        self.date_memory = [self.curData['date'].unique()[0]]
        self.reward_lst = [0]
        self.action_cbf_memeory = [np.array([0] * self.stock_num)]
        self.action_rl_memory = [np.array([1/self.stock_num]*self.stock_num) * self.bound_flag]
        self.risk_adj_lst = [cur_risk_boundary]
        self.is_last_ctrl_solvable = False
        self.risk_raw_lst = [0]
        self.risk_cbf_lst = [0]
        self.return_raw_lst = [self.initial_asset]
        self.solver_stat = {'solvable': 0, 'insolvable': 0, 'stochastic_solvable': 0, 'stochastic_time': [], 'socp_solvable': 0, 'socp_time': []} 
        self.ctrl_weight_lst = [1.0]
        self.solvable_flag = []
        self.risk_pred_lst = []
        self.rl_reward_risk_lst = []
        self.rl_reward_profit_lst = []
        self.cnt1 = 0
        self.cnt2 = 0
        self.stepcount = 0

        # Reset timing variables
        self.start_cputime = time.process_time()
        self.start_systime = time.perf_counter()
        
        return self.state

    def render(self, mode='human'):
        """Render the environment"""
        return self.state

    def save_action_memory(self):
        """Save action memory to dataframe"""
        return save_action_memory(self)

    def seed(self, seed=2022):
        """Set random seed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        """Get stable-baselines vector environment"""
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def get_results(self):
        """Calculate performance metrics"""
        info_dict, self.mdd, self.mdd_highTimepoint, self.mdd_lowTimepoint, self.mdd_high, self.mdd_low = calculate_portfolio_metrics(self)
        
        # Validate action memory shape
        if np.shape(np.array(self.actions_memory)) != (self.totalTradeDay, self.stock_num):
            if (self.config.mode =='RLcontroller') and (self.config.enable_controller):
                raise ValueError('actions_memory shape error in the RLcontroller mode')
            else:
                self.actions_memory = np.ones((self.totalTradeDay, self.stock_num)) * (1/self.stock_num) * self.bound_flag
                
        # Validate RL action memory shape
        if np.shape(np.array(self.action_rl_memory)) != (self.totalTradeDay+1, self.stock_num):
            if (self.config.mode =='RLcontroller') and (self.config.enable_controller):
                raise ValueError('action_rl_memory shape error in the RLcontroller mode')
            else:
                self.action_rl_memory = np.ones((self.totalTradeDay+1, self.stock_num)) * (1/self.stock_num) * self.bound_flag
                
        # Validate CBF action memory shape
        if np.shape(np.array(self.action_cbf_memeory)) != (self.totalTradeDay+1, self.stock_num):
            if (self.config.mode =='RLcontroller') and (self.config.enable_controller):
                raise ValueError('action_cbf_memeory shape error in the RLcontroller mode')
            else:
                self.action_cbf_memeory = np.zeros((self.totalTradeDay+1, self.stock_num))
                
        # Initialize empty arrays if needed
        if len(self.solvable_flag) == 0:
            self.solvable_flag = np.zeros(len(self.asset_lst))
        if len(self.risk_pred_lst) == 0:
            self.risk_pred_lst = np.zeros(len(self.asset_lst))
            
        return info_dict


class StockPortfolioEnv_cash(StockPortfolioEnv):
    """
    Stock portfolio trading environment with cash position.
    
    This environment extends the base environment to include a cash position.
    """
    def step(self, actions):
        """
        Take a step in the environment.
        
        Args:
            actions: Actions to take (portfolio weights)
            
        Returns:
            Tuple of (state, reward, done, info)
        """
        # Check if episode is over
        self.terminal = self.curTradeDay >= (self.totalTradeDay - 1)
        
        if self.terminal:
            # Apply transaction cost to final capital
            self.cur_capital = self.cur_capital * (1 - (np.sum(np.abs(self.actions_memory[-1])) * self.transaction_cost))
            self.asset_lst[-1] = self.cur_capital
            self.profit_lst[-1] = (self.cur_capital - self.asset_lst[-2]) / self.asset_lst[-2]     
            
            # Apply transaction cost to raw return without CBF
            if len(self.action_rl_memory) > 1:
                self.return_raw_lst[-1] = self.return_raw_lst[-1] * (1 - (np.sum(np.abs(self.action_rl_memory[-1])) * self.transaction_cost))

            # Train market observer at end of episode
            if (self.config.enable_market_observer) and (self.mode == 'train'):
                # Training at the end of epoch
                ori_profit_rate = np.append([1], np.array(self.return_raw_lst)[1:] / np.array(self.return_raw_lst)[:-1], axis=0)
                adj_profit_rate = np.array(self.profit_lst) + 1
                label_kwargs = {'ori_profit': ori_profit_rate, 'adj_profit': adj_profit_rate, 'ori_risk': np.array(self.risk_raw_lst), 'adj_risk': np.array(self.risk_cbf_lst)}
                self.mkt_observer.train(**label_kwargs)

            # Calculate elapsed time
            self.end_cputime = time.process_time()
            self.end_systime = time.perf_counter()
            self.model_save_flag = True
            
            # Calculate investment metrics
            invest_profile = self.get_results()
            save_profile(self, invest_profile)

            return self.state, self.reward, self.terminal, {}
        
        else:
            # Normalize actions to get portfolio weights
            actions = np.reshape(actions, (-1))  # [1, num_of_stocks] or [num_of_stocks, ]
            weights = self.weights_normalization(actions)  # Unnormalized weights -> normalized weights 
            self.actions_memory.append(weights[1:])  # Store stock weights without cash
            
            # Apply transaction costs
            if self.curTradeDay == 0:
                self.cur_capital = self.cur_capital * (1 - (1-1/len(weights)) * self.transaction_cost)
            else:
                # Calculate price changes with slippage
                cur_p = np.array(self.curData['close'].values) * (1 + self.cur_slippage_drift)
                last_p = np.array(self.lastDayData['close'].values) * (1 + self.last_slippage_drift)
                x_p = cur_p / last_p
                
                # Get last action and add cash position
                last_action = np.array(self.actions_memory[-2])
                last_action = np.append([1.0 - np.sum(np.abs(last_action))], last_action, axis=0)  # cash
                
                # Adjust for extreme price changes
                x_p_adj = np.where((x_p>=2)&(last_action[1:]<0), 2, x_p)
                x_p_adj = np.append([1.0], x_p_adj, axis=0)  # cash
                
                # Calculate signs and adjusted weights
                sgn = np.sign(last_action)
                sgn[0] = 1.0  # cash sign always positive
                adj_w_ay = sgn * (last_action * (x_p_adj - 1) + np.abs(last_action))
                adj_cap = np.sum((x_p_adj - 1) * last_action) + 1
                
                # Check for capital loss
                if (adj_cap <= 0) or np.all(adj_w_ay==0):
                    raise ValueError("Loss the whole capital! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(
                        self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay))
                
                # Normalize adjusted weights
                last_w_adj = adj_w_ay / adj_cap
                
                # Apply transaction costs
                self.cur_capital = self.cur_capital * (1 - (np.sum(np.abs(self.actions_memory[-1] - last_w_adj[1:])) * self.transaction_cost))
                self.asset_lst[-1] = self.cur_capital
                self.profit_lst[-1] = (self.cur_capital - self.asset_lst[-2]) / self.asset_lst[-2]
                
                # Calculate returns without CBF
                if len(self.action_rl_memory) > 1:
                    last_rl_action = np.array(self.action_rl_memory[-2])
                    last_rl_action = np.append([1.0 - np.sum(np.abs(last_rl_action))], last_rl_action, axis=0)  # cash
                    x_p_adjrl = np.where((x_p>=2)&(last_rl_action[1:]<0), 2, x_p)
                    sgn_rl = np.sign(last_rl_action)
                    sgn_rl[0] = 1.0  # cash sign always positive
                    prev_rl_cap = self.return_raw_lst[-1]
                    
                    # Adjust for extreme price changes in RL actions
                    adj_w_ay = sgn_rl * (last_rl_action * (x_p_adjrl - 1) + np.abs(last_rl_action))
                    adj_cap = np.sum((x_p_adjrl - 1) * last_rl_action) + 1
                    
                    # Check for capital loss without CBF
                    if (adj_cap <= 0) or np.all(adj_w_ay==0):
                        print("Loss the whole capital if using RL actions only! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(
                            self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay))
                        adj_w_ay = np.array([1/(self.stock_num+1)]*(self.stock_num+1)) * self.bound_flag
                        adj_cap = 1
                    
                    # Normalize adjusted weights for RL
                    last_rlw_adj = adj_w_ay / adj_cap
                    
                    # Apply transaction costs to raw return
                    return_raw = prev_rl_cap * (1 - (np.sum(np.abs(self.action_rl_memory[-1] - last_rlw_adj[1:])) * self.transaction_cost))
                    self.return_raw_lst[-1] = return_raw       
            
            # Advance to next day
            self.curTradeDay = self.curTradeDay + 1
            self.lastDayData = self.curData       
            self.last_slippage_drift = self.cur_slippage_drift     
            
            # Load new day's data
            self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
            self.curData.sort_values(['stock'], ascending=True, inplace=True)
            self.curData.reset_index(drop=True, inplace=True)
            
            # Update state
            if self.config.enable_cov_features:   
                self.covs = np.array(self.curData['cov'].values[0])
                self.state = np.append(self.covs, np.transpose(self.curData[self.tech_indicator_lst_wocov].values), axis=0)
            else:
                self.state = np.transpose(self.curData[self.tech_indicator_lst_wocov].values)
            self.state = self.state.flatten()
            
            # Update controller state
            self.ctl_state = {k:np.array(list(self.curData[k].values)) for k in self.config.otherRef_indicator_lst}
            
            # Record date
            cur_date = self.curData['date'].unique()[0]
            self.date_memory.append(cur_date)

            # Generate new slippage drift
            self.cur_slippage_drift = np.random.random(self.stock_num) * (self.slippage * 2) - self.slippage
            
            # Calculate price changes with slippage
            curDay_ClosePrice_withSlippage = np.array(self.curData['close'].values) * (1 + self.cur_slippage_drift)
            lastDay_ClosePrice_withSlippage = np.array(self.lastDayData['close'].values) * (1 + self.last_slippage_drift)
            rate_of_price_change = curDay_ClosePrice_withSlippage / lastDay_ClosePrice_withSlippage
            
            # Adjust for extreme price changes
            rate_of_price_change_adj = np.where((rate_of_price_change>=2)&(weights[1:]<0), 2, rate_of_price_change)
            
            # Calculate single day returns
            sigDayReturn = (rate_of_price_change_adj - 1) * weights[1:]  # [s1_pct, s2_pct, .., px_pct_returns]
            poDayReturn = np.sum(sigDayReturn)
            
            # Check for capital loss
            if poDayReturn <= (-1):
                raise ValueError("Loss the whole capital! [Day: {}, date: {}, poDayReturn: {}]".format(
                    self.curTradeDay, self.date_memory[-1], poDayReturn))
            
            # Update portfolio value with cash component
            updatePoValue = self.cur_capital * ((poDayReturn + 1 - np.abs(weights[0])) + np.abs(weights[0]))
            poDayReturn_withcost = (updatePoValue - self.cur_capital) / self.cur_capital  
            
            self.cur_capital = updatePoValue
            self.state = np.append(self.state, [np.log((self.cur_capital/self.initial_asset))], axis=0)
            
            # Record daily returns and portfolio value
            self.profit_lst.append(poDayReturn_withcost) 
            self.asset_lst.append(self.cur_capital)

            # Get market observer info including cash
            rate_of_price_change_withcash = np.append([1.0], rate_of_price_change_adj, axis=0)  # cash
            cur_risk_boundary, stock_ma_price = run_market_observer(
                self, stage='run', rate_of_price_change=np.array([rate_of_price_change_withcash])
            )
            
            # Update MA prices if available
            if stock_ma_price is not None:
                self.ctl_state['MA-{}'.format(self.config.otherRef_indicator_ma_window)] = stock_ma_price
                
            # Record risk adjustments
            self.risk_adj_lst.append(cur_risk_boundary)
            self.ctrl_weight_lst.append(1.0) 

            # Calculate risks
            daily_return_ay = np.array(list(self.curData['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)].values))
            cur_cov = np.cov(daily_return_ay) 
            self.risk_cbf_lst.append(np.sqrt(np.matmul(np.matmul(weights[1:], cur_cov), weights[1:].T)))
            
            # Calculate risk without CBF
            w_rl = self.action_rl_memory[-1]
            w_rl = w_rl / np.sum(np.abs(w_rl))
            self.risk_raw_lst.append(np.sqrt(np.matmul(np.matmul(w_rl, cur_cov), w_rl.T)))

            # Calculate returns without CBF with cash component
            if self.curTradeDay == 1:
                prev_rl_cap = self.return_raw_lst[-1] * (1 - (1-1/len(weights)) * self.transaction_cost)
            else:
                prev_rl_cap = self.return_raw_lst[-1]

            # Adjust for extreme price changes in RL actions
            rate_of_price_change_adj_rawrl = np.where((rate_of_price_change>=2)&(w_rl<0), 2, rate_of_price_change)
            
            # Calculate return with cash
            return_raw = prev_rl_cap * ((np.sum((rate_of_price_change_adj_rawrl - 1) * w_rl) + 1 - np.abs(weights[0])) + np.abs(weights[0]))
            
            # Check for capital loss
            if return_raw <= 0:
                raise ValueError("Loss the whole capital if using RL actions only! [Day: {}, date: {}, return_raw: {}]".format(
                    self.curTradeDay, self.date_memory[-1], return_raw))
                
            self.return_raw_lst.append(return_raw)

            # Calculate CVaR (Conditional Value at Risk)
            expected_r_series = daily_return_ay[:, -21:]
            expected_r_prev = np.mean(expected_r_series[:, -1:], axis=1)
            expected_r_prev = np.where((expected_r_prev>=1)&(weights[1:]<0), 1, expected_r_prev)
            expected_r = np.sum(np.reshape(expected_r_prev, (1, -1)) @ np.reshape(weights[1:], (-1, 1)))
            expected_cov = np.cov(expected_r_series)
            expected_std = np.sum(np.sqrt(np.reshape(weights[1:], (1, -1)) @ expected_cov @ np.reshape(weights[1:], (-1, 1))))
            cvar_lz = spstats.norm.ppf(1-0.05)  # positive 1.65 for 95%(=1-alpha) confidence level
            cvar_Z = np.exp(-0.5*np.power(cvar_lz, 2)) / 0.05 / np.sqrt(2*np.pi)
            cvar_expected = -expected_r + expected_std * cvar_Z
            self.cvar_lst.append(cvar_expected)

            # Calculate CVaR without risk controller
            expected_r_prevrl = np.mean(expected_r_series[:, -1:], axis=1)
            expected_r_prevrl = np.where((expected_r_prevrl>=1)&(w_rl<0), 1, expected_r_prevrl)
            expected_r_raw = np.sum(np.reshape(expected_r_prevrl, (1, -1)) @ np.reshape(w_rl, (-1, 1)))
            expected_std_raw = np.sum(np.sqrt(np.reshape(w_rl, (1, -1)) @ expected_cov @ np.reshape(w_rl, (-1, 1))))
            cvar_expected_raw = -expected_r_raw + expected_std_raw * cvar_Z
            self.cvar_raw_lst.append(cvar_expected_raw)

            # Calculate reward considering cash position
            profit_part = np.log(poDayReturn_withcost+1)
            
            # Different reward calculations based on model type
            if (self.config.trained_best_model_type == 'js_loss') and (self.config.enable_controller):
                # Jensen-Shannon divergence reward with cash position
                if self.config.trade_pattern == 1:
                    # Long only
                    weights_norm = weights
                    w_rl_norm = w_rl
                elif self.config.trade_pattern == 2:
                    # Long-short, normalize to [0,1]
                    weights_norm = (weights + 1) / 2
                    w_rl_norm = (w_rl + 1) / 2
                elif self.config.trade_pattern == 3:
                    # Short only, normalize to [0,1]
                    weights_norm = -weights
                    w_rl_norm = -w_rl
                else:
                    raise ValueError("Unexpected trade pattern: {}".format(self.config.trade_pattern))
                
                # Calculate JS divergence between RL weights and actual weights
                js_m = 0.5 * (w_rl_norm + weights_norm[1:])
                js_divergence = (0.5 * entropy(pk=w_rl_norm, qk=js_m, base=2)) + (0.5 * entropy(pk=weights_norm[1:], qk=js_m, base=2))
                js_divergence = np.clip(js_divergence, 0, 1)
                risk_part = (-1) * js_divergence

                scaled_risk_part = self.config.lambda_2 * risk_part
                scaled_profit_part = self.config.lambda_1 * profit_part         
                self.reward = scaled_profit_part + scaled_risk_part      

            elif (self.config.mode == 'RLonly') and (self.config.trained_best_model_type == 'pr_loss'):
                # Profit-risk reward
                cov_r_t0 = np.cov(self.ctl_state['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)])
                risk_part = np.sqrt(np.matmul(np.matmul(np.array([weights[1:]]), cov_r_t0), np.array([weights[1:]]).T)[0][0])
                scaled_risk_part = (-1) * risk_part * 50
                scaled_profit_part = profit_part * self.config.lambda_1
                self.reward = scaled_profit_part + scaled_risk_part

            elif (self.config.mode == 'RLonly') and (self.config.trained_best_model_type == 'sr_loss'):
                # Sharpe ratio reward              
                cov_r_t0 = np.cov(self.ctl_state['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)])
                risk_part = np.sqrt(np.matmul(np.matmul(np.array([weights[1:]]), cov_r_t0), np.array([weights[1:]]).T)[0][0])
                profit_part = poDayReturn_withcost
                scaled_profit_part = profit_part
                scaled_risk_part = risk_part
                self.reward = (scaled_profit_part - (self.config.mkt_rf[self.config.market_name] * 0.01)) / scaled_risk_part

            else:
                # Default profit-only reward
                risk_part = 0
                scaled_risk_part = 0
                scaled_profit_part = profit_part * self.config.lambda_1
                self.reward = scaled_profit_part + scaled_risk_part

            # Record reward components
            self.rl_reward_risk_lst.append(scaled_risk_part)
            self.rl_reward_profit_lst.append(scaled_profit_part)
            self.reward_lst.append(self.reward)
            self.model_save_flag = False

            return self.state, self.reward, self.terminal, {}