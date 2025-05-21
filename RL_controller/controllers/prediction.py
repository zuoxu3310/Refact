"""
价格预测相关功能。
"""
import numpy as np


def get_pred_price_change(env):
    """
    获取预测的价格变化。
    使用移动平均(MA)模型预测价格变化。
    
    :param env: 环境对象
    :return: 预测的价格变化率
    """
    # 获取移动平均价格
    ma_lst = env.ctl_state[f'MA-{env.config.otherRef_indicator_ma_window}']
    pred_prices = ma_lst
    
    # 计算当前收盘价
    cur_close_price = np.array(env.curData['close'].values)
    
    # 计算预测价格变化率
    pred_prices_change = (pred_prices - cur_close_price) / cur_close_price 
    
    return pred_prices_change