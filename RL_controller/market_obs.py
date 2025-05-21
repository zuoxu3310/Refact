#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name:         market_obs.py
 Description:  Implement the market observer of the proposed MASA framework.
 Author:       MASA
---------------------------------
'''
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Callable, Any

# 导入重构后的所有组件
from .market_obs.registry import (
    register_mkt_obs_model,
    is_model,
    mkt_obs_model_entrypoint,
    create_mkt_obs_model,
    _mkt_obs_model_entrypoints
)
from .market_obs.observers import (
    MarketObserver,
    MarketObserver_Algorithmic
)
from .market_obs.models.common import GenScore
from .market_obs.models.mlp import MLP_1, STF_1
from .market_obs.models.lstm import LSTM_1
from .market_obs.models.algorithmic import MA_1, DC_1

# 为了兼容性，将必要的类和函数重新导出到全局命名空间
th.autograd.set_detect_anomaly(True)

# 保留原始导出的名称，以保持向后兼容性
__all__ = [
    'register_mkt_obs_model',
    'is_model',
    'mkt_obs_model_entrypoint',
    'create_mkt_obs_model',
    'MarketObserver',
    'MarketObserver_Algorithmic',
    'GenScore',
    'MLP_1',
    'LSTM_1',
    'STF_1',
    'MA_1',
    'DC_1',
    'mlp_1',
    'lstm_1',
    'stf_1',
    'ma_1',
    'dc_1',
]