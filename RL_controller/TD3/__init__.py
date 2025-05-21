"""
基于Twin Delayed DDPG (TD3)的控制器实现。
"""

# 从各子模块导入公共API
from .utils import create_mlp_adj
from .actors import ActorAdj, ActorOriginal
from .policies import TD3PolicyAdj, TD3PolicyOriginal
from .controller import TD3Controller

# 导出公共API
__all__ = [
    "create_mlp_adj",
    "ActorAdj",
    "ActorOriginal",
    "TD3PolicyAdj", 
    "TD3PolicyOriginal",
    "TD3Controller"
]