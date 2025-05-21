"""
MASA框架的市场观察者模块。
提供了不同的市场观察者实现和模型。
"""

# 从各子模块导入公共API
from .registry import (
    register_mkt_obs_model,
    is_model,
    mkt_obs_model_entrypoint,
    create_mkt_obs_model
)
from .observers import (
    MarketObserver,
    MarketObserver_Algorithmic
)

# 导入所有模型以触发注册机制
from .models import *

__all__ = [
    'register_mkt_obs_model',
    'is_model',
    'mkt_obs_model_entrypoint',
    'create_mkt_obs_model',
    'MarketObserver',
    'MarketObserver_Algorithmic',
]