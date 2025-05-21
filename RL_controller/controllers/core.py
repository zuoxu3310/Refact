"""
MASA框架中控制器的核心功能。
包含控制器与RL组合的策略实现。
"""
import numpy as np

from .prediction import get_pred_price_change
from .optimization import cbf_opt


def RL_withoutController(a_rl, env=None):
    """
    不使用控制器的RL决策。
    
    :param a_rl: RL代理产生的动作
    :param env: 环境对象
    :return: 最终动作（不含控制器修正）
    """
    a_cbf = np.array([0] * env.stock_num)
    a_rl = np.array(a_rl)
    env.action_cbf_memeory.append(a_cbf)
    env.action_rl_memory.append(a_rl)
    a_final = a_rl + a_cbf
    return a_final


def RL_withController(a_rl, env=None):
    """
    使用控制器的RL决策。
    RL动作与控制器产生的修正相结合。
    
    :param a_rl: RL代理产生的动作
    :param env: 环境对象
    :return: 最终修正后的动作
    """
    a_rl = np.array(a_rl)
    env.action_rl_memory.append(a_rl)
    
    # 根据配置选择价格预测模型
    if env.config.pricePredModel == 'MA':
        pred_prices_change = get_pred_price_change(env=env)
        pred_dict = {'shortterm': pred_prices_change}
    else:
        raise ValueError(f"Cannot find the price prediction model [{env.config.pricePredModel}]..")
    
    # 使用CBF优化计算控制器修正
    a_cbf, is_solvable_status = cbf_opt(env=env, a_rl=a_rl, pred_dict=pred_dict) 
    
    # 设置权重
    cur_dcm_weight = 1.0 
    cur_rl_weight = 1.0
    
    if is_solvable_status:   
        # 如果优化问题可解，应用控制器修正
        a_cbf_weighted = a_cbf * cur_dcm_weight
        env.action_cbf_memeory.append(a_cbf_weighted)
        a_rl_weighted = a_rl * cur_rl_weight 
        a_final = a_rl_weighted + a_cbf_weighted
    else:
        # 如果优化问题不可解，使用原始RL动作
        env.action_cbf_memeory.append(np.array([0] * env.stock_num))
        a_final = a_rl
        
    return a_final