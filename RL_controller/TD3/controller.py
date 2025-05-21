"""
Twin Delayed DDPG (TD3)控制器实现。
"""
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import gymnasium
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TrainFreq, TrainFrequencyUnit, RolloutReturn
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, should_collect_more_steps
from stable_baselines3.td3.policies import MlpPolicy, CnnPolicy, MultiInputPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from .policies import TD3PolicyAdj, TD3PolicyOriginal
from ..controllers.core import RL_withController

SelfTD3 = TypeVar("SelfTD3", bound="TD3Controller")


class TD3Controller(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)，支持控制器集成。
    解决Actor-Critic方法中的函数近似误差问题。
    原始实现: https://github.com/sfujim/TD3
    论文: https://arxiv.org/abs/1802.09477
    TD3介绍: https://spinningup.openai.com/en/latest/algorithms/td3.html
    
    :param policy: 要使用的策略模型(MlpPolicy, CnnPolicy, ...)
    :param env: 学习环境(如果在Gym中注册，可以是str)
    :param learning_rate: adam优化器的学习率，
        所有网络使用相同的学习率(Q-Values, Actor和Value函数)
        可以是当前剩余进度的函数(从1到0)
    :param buffer_size: 重放缓冲区大小
    :param learning_starts: 开始学习前收集多少步过渡
    :param batch_size: 每次梯度更新的小批量大小
    :param tau: 软更新系数("Polyak更新", 在0和1之间)
    :param gamma: 折扣因子
    :param train_freq: 每 train_freq 步更新模型。
                      或者传递频率和单位的元组如 (5, "step") 或 (2, "episode")
    :param gradient_steps: 每次rollout后执行多少梯度步骤(见train_freq)
                          设置为-1表示执行与rollout期间环境中执行的步骤相同数量的梯度步骤
    :param action_noise: 行动噪声类型(默认为None)，这可以帮助
                       解决难探索问题。参见common.noise了解不同的行动噪声类型
    :param replay_buffer_class: 使用的重放缓冲区类(例如HerReplayBuffer)
                              如果为None，将自动选择
    :param replay_buffer_kwargs: 创建重放缓冲区时传递的关键字参数
    :param optimize_memory_usage: 启用重放缓冲区的内存高效变体，
                                代价是更高的复杂性
    :param policy_delay: 每训练步骤的policy_delay步数才会更新策略和目标网络
                       Q值将比策略更新更频繁(每训练步骤更新)
    :param target_policy_noise: 添加到目标策略的高斯噪声标准差(平滑噪声)
    :param target_noise_clip: 目标策略平滑噪声绝对值的限制
    :param policy_kwargs: 创建策略时传递的额外参数
    :param verbose: 详细程度: 0表示无输出, 1表示信息消息(如使用的设备或包装器), 2表示调试消息
    :param seed: 伪随机生成器的种子
    :param device: 代码应该运行的设备(cpu, cuda, ...)
                  设置为auto，代码将在GPU上运行(如果可能)
    :param _init_setup_model: 是否在创建实例时构建网络
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
        'TD3PolicyAdj': TD3PolicyAdj,
        'TD3PolicyOriginal': TD3PolicyOriginal,
    }

    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gymnasium.spaces.Box,),
            support_multi_env=True,
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """设置模型的内部组件。"""
        super()._setup_model()
        self._create_aliases()
        # 运行均值和方差
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        """创建常用组件的别名。"""
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        训练TD3模型。
        
        :param gradient_steps: 要执行的梯度步骤数量
        :param batch_size: 批次大小
        """
        # 切换到训练模式(这会影响batch norm / dropout)
        self.policy.set_training_mode(True)

        # 根据学习率调度更新学习率
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):

            self._n_updates += 1
            # 从重放缓冲区采样
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # 根据策略选择动作并添加裁剪噪声
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # 计算下一个Q值：所有评论家目标的最小值
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # 获取每个评论家网络的当前Q值估计
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # 计算评论家损失
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # 优化评论家
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # 延迟策略更新
            if self._n_updates % self.policy_delay == 0:
                # 计算演员损失
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # 优化演员
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # 复制运行统计信息，参见GH问题#996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self: SelfTD3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTD3:
        """
        训练模型的主要入口点。
        
        :param total_timesteps: 训练的总时间步数
        :param callback: 在训练过程中调用的回调
        :param log_interval: 日志记录的间隔
        :param tb_log_name: tensorboard日志的名称
        :param reset_num_timesteps: 是否重置时间步计数器
        :param progress_bar: 是否显示进度条
        :return: 训练后的模型实例
        """
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        """指定保存模型时要排除的参数。"""
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        """获取torch保存参数。"""
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        收集经验并存储到重放缓冲区中。
        
        :param env: 训练环境
        :param callback: 在每一步调用的回调(以及rollout的开始和结束)
        :param train_freq: 通过执行当前策略的rollouts收集多少经验
        :param action_noise: 用于探索的动作噪声
        :param learning_starts: 热身阶段学习前的步数
        :param replay_buffer: 重放缓冲区
        :param log_interval: 每多少个episode记录日志
        :return: RolloutReturn对象
        """
        # 切换到评估模式(这会影响batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # 如果需要，向量化动作噪声
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)
        
        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # 采样新的噪声矩阵
                self.actor.reset_noise(env.num_envs)

            # 随机选择动作或根据策略选择
            # actions: 范围-[low, high], 形状: [1, num_of_stocks], buffer_actions: [-1, 1] 用于actor和critic代理训练
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            a_rlonly = np.array(actions[0]) # [1, num_of_stocks] -> [num_of_stocks, ]
            a_rl = a_rlonly
            if np.sum(np.abs(a_rl)) == 0:
                a_rl = np.array([1/len(a_rl)]*len(a_rl)) * env.envs[0].bound_flag
            else:
                a_rl = a_rl / np.sum(np.abs(a_rl))

            a_final = RL_withController(a_rl=a_rl, env=env.envs[0].env)
            a_final = a_final / np.sum(np.abs(a_final))

            # 重新缩放并执行动作
            a_final = np.array([a_final])
            new_obs, rewards, dones, infos = env.step(a_final)
            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # 提供对本地变量的访问
            callback.update_locals(locals())
            # 只有在返回值为False时才停止训练，而不是为None时
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # 如果使用Monitor包装器，检索奖励和episode长度
            self._update_info_buffer(infos, dones)

            # 将数据存储在重放缓冲区中(归一化的动作和未归一化的观察)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # 对于DQN，检查是否应更新目标网络
            # 并更新探索计划
            # 对于SAC/TD3，更新与梯度更新同时进行
            # 参见 https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # 更新统计信息
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # 记录训练信息
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)