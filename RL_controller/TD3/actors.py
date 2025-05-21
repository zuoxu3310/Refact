"""
TD3控制器的Actor网络实现。
"""
from typing import Any, Dict, List, Type
import gym
import torch as th
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim

from .utils import create_mlp_adj


class ActorAdj(BasePolicy):
    """
    TD3的Actor网络(策略)。
    
    :param observation_space: 观察空间
    :param action_space: 动作空间
    :param net_arch: 网络架构
    :param features_extractor: 特征提取网络
        (使用图像时为CNN，否则为nn.Flatten()层)
    :param features_dim: 特征数量
    :param activation_fn: 激活函数
    :param normalize_images: 是否归一化图像，
         除以255.0(默认为True)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: th.nn.Module,
        features_dim: int,
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        self.action_dim = action_dim
        # actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        td3_main_branch_dim = features_dim - action_dim
        actor_net = create_mlp_adj(td3_main_branch_dim, action_dim, net_arch, activation_fn, squash_output=True)

        # 确定性动作
        self.mu = th.nn.Sequential(*actor_net)
        # self.mkt_hidden_sm = th.nn.Softmax(dim=1) # 对市场观察者执行softmax。

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs, self.features_extractor)
        td3_decision = self.mu(features[:, :-self.action_dim]) # 范围 [0, 1], 和为1
        mkt_decision = features[:, -self.action_dim:] # 范围 [0, 1], 和为1
        final_output = (td3_decision + mkt_decision) - 1 # 范围 [-1, 1]
        return final_output

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # 注意：TD3的情况下deterministic参数被忽略。
        # 预测总是确定性的。
        return self(observation)


class ActorOriginal(BasePolicy):
    """
    TD3的Actor网络(策略)。
    
    :param observation_space: 观察空间
    :param action_space: 动作空间
    :param net_arch: 网络架构
    :param features_extractor: 特征提取网络
        (使用图像时为CNN，否则为nn.Flatten()层)
    :param features_dim: 特征数量
    :param activation_fn: 激活函数
    :param normalize_images: 是否归一化图像，
         除以255.0(默认为True)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: th.nn.Module,
        features_dim: int,
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        self.action_dim = action_dim
        # actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        actor_net = create_mlp_adj(features_dim, action_dim, net_arch, activation_fn, squash_output=True)

        # 确定性动作
        self.mu = th.nn.Sequential(*actor_net)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        final_output = self.mu(features) # 范围 [0, 1], 和为1
        return final_output

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # 注意：TD3的情况下deterministic参数被忽略。
        # 预测总是确定性的。
        return self(observation)