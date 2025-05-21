"""
TD3策略实现，支持调整行为。
"""
from typing import Optional
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy

from .actors import ActorAdj, ActorOriginal


class TD3PolicyAdj(TD3Policy):
    """
    带有调整的Actor实现的TD3策略。
    """
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ActorAdj:
        """
        创建调整版Actor。
        
        :param features_extractor: 特征提取器
        :return: ActorAdj实例
        """
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ActorAdj(**actor_kwargs).to(self.device)


class TD3PolicyOriginal(TD3Policy):
    """
    带有原始Actor实现的TD3策略。
    """
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ActorOriginal:
        """
        创建原始版Actor。
        
        :param features_extractor: 特征提取器
        :return: ActorOriginal实例
        """
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ActorOriginal(**actor_kwargs).to(self.device)