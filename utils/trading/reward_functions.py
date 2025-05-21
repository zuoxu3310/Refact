#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         reward_functions.py
 Description:  Reward functions for trading environment.
 Author:       MASA
---------------------------------
'''
import numpy as np
from scipy.stats import entropy

def calculate_reward(env, weights, w_rl, poDayReturn_withcost):
    """
    Calculate reward based on the current model type.
    
    Args:
        env: Trading environment instance
        weights: Current action weights
        w_rl: RL action weights without CBF adjustment
        poDayReturn_withcost: Portfolio daily return with transaction cost
        
    Returns:
        Tuple of (reward, profit_part, risk_part, scaled_profit_part, scaled_risk_part)
    """
    profit_part = np.log(poDayReturn_withcost + 1)
    
    if (env.config.trained_best_model_type == 'js_loss') and (env.config.enable_controller):
        # Jensen-Shannon divergence as reward
        risk_part, scaled_profit_part, scaled_risk_part = js_divergence_reward(
            env, weights, w_rl, profit_part
        )
    elif (env.config.mode == 'RLonly') and (env.config.trained_best_model_type == 'pr_loss'):
        # Profit-Risk reward
        risk_part, scaled_profit_part, scaled_risk_part = profit_risk_reward(
            env, weights, profit_part
        )
    elif (env.config.mode == 'RLonly') and (env.config.trained_best_model_type == 'sr_loss'):
        # Sharpe ratio reward
        risk_part, scaled_profit_part, scaled_risk_part = sharpe_ratio_reward(
            env, weights, poDayReturn_withcost
        )
    else:
        # Default profit-only reward
        risk_part = 0
        scaled_risk_part = 0
        scaled_profit_part = profit_part * env.config.lambda_1
    
    reward = scaled_profit_part + scaled_risk_part
    
    return reward, profit_part, risk_part, scaled_profit_part, scaled_risk_part

def js_divergence_reward(env, weights, w_rl, profit_part):
    """Calculate Jensen-Shannon divergence-based reward"""
    # Normalize weights based on trade pattern
    if env.config.trade_pattern == 1:
        # Long only
        weights_norm = weights
        w_rl_norm = w_rl
    elif env.config.trade_pattern == 2:
        # Long-short, normalize to [0,1]
        weights_norm = (weights + 1) / 2
        w_rl_norm = (w_rl + 1) / 2
    elif env.config.trade_pattern == 3:
        # Short only, normalize to [0,1]
        weights_norm = -weights
        w_rl_norm = -w_rl
    else:
        raise ValueError("Unexpected trade pattern: {}".format(env.config.trade_pattern))
    
    # Calculate Jensen-Shannon divergence
    js_m = 0.5 * (w_rl_norm + weights_norm)
    js_divergence = (0.5 * entropy(pk=w_rl_norm, qk=js_m, base=2)) + (0.5 * entropy(pk=weights_norm, qk=js_m, base=2))
    js_divergence = np.clip(js_divergence, 0, 1)
    
    # Risk part is negative JS divergence (lower is better)
    risk_part = (-1) * js_divergence
    scaled_profit_part = env.config.lambda_1 * profit_part
    scaled_risk_part = env.config.lambda_2 * risk_part
    
    return risk_part, scaled_profit_part, scaled_risk_part

def profit_risk_reward(env, weights, profit_part):
    """Calculate profit-risk reward"""
    # Get covariance matrix for risk calculation
    cov_r_t0 = np.cov(env.ctl_state['DAILYRETURNS-{}'.format(env.config.dailyRetun_lookback)])
    
    # Calculate portfolio risk
    if hasattr(weights, 'shape') and len(weights.shape) > 1:
        risk_part = np.sqrt(np.matmul(np.matmul(weights, cov_r_t0), weights.T)[0][0])
    else:
        risk_part = np.sqrt(np.matmul(np.matmul(np.array([weights]), cov_r_t0), np.array([weights]).T)[0][0])
    
    # Scale risk part (negative as we want to minimize risk)
    scaled_risk_part = (-1) * risk_part * 50
    scaled_profit_part = profit_part * env.config.lambda_1
    
    return risk_part, scaled_profit_part, scaled_risk_part

def sharpe_ratio_reward(env, weights, poDayReturn_withcost):
    """Calculate Sharpe ratio reward"""
    # Get covariance matrix for risk calculation
    cov_r_t0 = np.cov(env.ctl_state['DAILYRETURNS-{}'.format(env.config.dailyRetun_lookback)])
    
    # Calculate portfolio risk
    if hasattr(weights, 'shape') and len(weights.shape) > 1:
        risk_part = np.sqrt(np.matmul(np.matmul(weights, cov_r_t0), weights.T)[0][0])
    else:
        risk_part = np.sqrt(np.matmul(np.matmul(np.array([weights]), cov_r_t0), np.array([weights]).T)[0][0])
    
    # Calculate profit part (actual return)
    profit_part = poDayReturn_withcost
    scaled_profit_part = profit_part
    scaled_risk_part = risk_part
    
    # Sharpe ratio = (return - risk_free_rate) / volatility
    reward = (scaled_profit_part - (env.config.mkt_rf[env.config.market_name] * 0.01)) / scaled_risk_part
    
    return risk_part, scaled_profit_part, scaled_risk_part