"""
市场观察者模型的集合。
"""

# 导入所有模型以触发注册机制
from .mlp import *
from .lstm import *
from .algorithmic import *
from .common import *

__all__ = [
    # MLP模型
    'mlp_1',
    # LSTM模型
    'lstm_1',
    # 其他模型
    'stf_1',
    # 算法模型
    'ma_1',
    'dc_1',
    # 通用组件
    'GenScore',
]