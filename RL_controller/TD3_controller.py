"""
Twin Delayed DDPG (TD3)控制器实现。
"""
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import gymnasium
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TrainFreq, TrainFrequencyUnit, RolloutReturn 
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.td3.policies import MlpPolicy, CnnPolicy, MultiInputPolicy

# 使用相对导入正确引用模块
from .TD3.utils import create_mlp_adj
from .TD3.actors import ActorAdj, ActorOriginal
from .TD3.policies import TD3PolicyAdj, TD3PolicyOriginal
from .TD3.controller import TD3Controller

# 为保持向后兼容性，将所有必要的组件导出到全局命名空间
__all__ = [
    "create_mlp_adj",
    "ActorAdj",
    "ActorOriginal",
    "TD3PolicyAdj", 
    "TD3PolicyOriginal",
    "TD3Controller",
]