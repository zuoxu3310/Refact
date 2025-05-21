"""
市场观察者的实现。
"""
import numpy as np
import torch as th
import torch.optim as optim

from .registry import create_mkt_obs_model

class MarketObserver_Algorithmic:
    """
    算法市场观察者类。
    这是一个简化版的市场观察者，主要用于算法交易策略。
    """
    def __init__(self, config, action_dim):
        """
        初始化算法市场观察者。
        
        :param config: 配置对象
        :param action_dim: 动作空间维度
        """
        self.config = config
        self.action_dim = action_dim  # 隐藏向量的维度
        input_kwargs = {'action_dim': self.action_dim}
        self.mkt_obs_model = create_mkt_obs_model(config=self.config, **input_kwargs)
    
    def train(self, **label_kwargs):
        """训练模型（占位方法）"""
        pass
    
    def reset(self):
        """重置模型状态（占位方法）"""
        pass
    
    def predict(self, finemkt_feat, finestock_feat, **kwargs):
        """
        预测市场行为。
        
        :param finemkt_feat: 市场特征
        :param finestock_feat: 股票特征
        :param kwargs: 其他参数
        :return: (隐藏向量，lambda值，sigma值)
        """
        cur_hidden_vector_ay, lambda_val_ay, sigma_val_ay = self.mkt_obs_model(**kwargs)
        return cur_hidden_vector_ay, lambda_val_ay, sigma_val_ay
    
    def update_hidden_vec_reward(self, mode, rate_of_price_change, mkt_direction):
        """
        更新隐藏向量奖励（占位方法）。
        
        :param mode: 模式
        :param rate_of_price_change: 价格变化率
        :param mkt_direction: 市场方向
        """
        pass


class MarketObserver:
    """
    完整的市场观察者类，包含训练和预测功能。
    """
    def __init__(self, config, action_dim):
        """
        初始化市场观察者。
        
        :param config: 配置对象
        :param action_dim: 动作空间维度
        """
        self.config = config
        self.action_dim = action_dim  # 隐藏向量的维度
        input_kwargs = {'action_dim': self.action_dim}
        self.mkt_obs_model = create_mkt_obs_model(config=self.config, **input_kwargs)

        # 设置设备
        if th.cuda.is_available():
            cuda_status = 'cuda:0'
        else:
            cuda_status = 'cpu'
        self.device = th.device(cuda_status)
        
        # 将模型移动到设备
        isParallel = False
        self.mkt_obs_model = self.mkt_obs_model.to(self.device)
        
        # 设置优化器
        if isParallel:
            self.optimizer = optim.Adam(self.mkt_obs_model.module.parameters(), 
                                        lr=self.config.po_lr, 
                                        weight_decay=self.config.po_weight_decay)
        else:
            self.optimizer = optim.Adam(self.mkt_obs_model.parameters(), 
                                        lr=self.config.po_lr, 
                                        weight_decay=self.config.po_weight_decay)

        # 设置学习率调度器
        decay_steps = self.config.num_epochs // 3
        self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_steps, gamma=0.1)

        # 初始化存储列表
        self.cur_hidden_vector_lst = []
        self.hidden_vector_reward_lst = []
        self.lambda_log_p_lst = []
        self.sigma_log_p_lst = []
        self.mkt_direction_lst = []

        # 损失函数
        self.mkt_direction_loss_sigma = th.nn.CrossEntropyLoss()
        self.mkt_direction_loss_lambda = th.nn.CrossEntropyLoss()

    def train(self, **label_kwargs):
        """
        训练市场观察者模型。
        
        :param label_kwargs: 标签参数
        """
        self.mkt_obs_model.train()
        
        # 计算损失
        hidden_vector_reward_tensor = th.cat(self.hidden_vector_reward_lst, dim=0)  # (all_samples, )
        loss_hidden = -th.mean(hidden_vector_reward_tensor)  # (1, )
        loss_val = self.config.hidden_vec_loss_weight * loss_hidden
        disp_str = f'Loss(Hidden): {self.config.hidden_vec_loss_weight * loss_hidden.detach().cpu().item()} |'

        if self.config.is_enable_dynamic_risk_bound:
            # sigma损失
            sigma_log_p_tensor = th.cat(self.sigma_log_p_lst, dim=0)  # (all_samples, )
            mkt_direction_tensor = th.cat(self.mkt_direction_lst, dim=0)  # (all_samples, )
            loss_sigma = self.mkt_direction_loss_sigma(sigma_log_p_tensor[:-1], mkt_direction_tensor)
            loss_val = loss_val + loss_sigma
            disp_str = disp_str + f'Loss(Sigma): {self.config.sigma_loss_weight * loss_sigma.detach().cpu().item()} |'

        # 反向传播
        self.optimizer.zero_grad()
        loss_val.backward()
        th.cuda.synchronize()
        self.optimizer.step()
        self.exp_lr_scheduler.step()

        # 显示训练信息
        disp_str = f'{label_kwargs["mode"]} | Loss(Total): {loss_val.detach().cpu().item()} |' + disp_str
        print(disp_str)

        # 重置状态
        self.reset()
        th.cuda.empty_cache()

    def reset(self):
        """重置模型状态和存储列表"""
        self.cur_hidden_vector_lst = []
        self.hidden_vector_reward_lst = []
        self.lambda_log_p_lst = []
        self.sigma_log_p_lst = []
        self.mkt_direction_lst = []

    def predict(self, finemkt_feat, finestock_feat, **kwargs):
        """
        预测市场行为。
        
        :param finemkt_feat: 市场特征 (batch, features, window_size)
        :param finestock_feat: 股票特征 (batch, features, num_of_stocks, window_size)
        :param kwargs: 其他参数，必须包含'mode'
        :return: (隐藏向量，lambda值，sigma值)
        """
        # 将numpy数组转换为张量
        finemkt_feat = th.from_numpy(finemkt_feat).to(th.float32)
        finestock_feat = th.from_numpy(finestock_feat).to(th.float32)

        # 移动到设备
        finemkt_feat = finemkt_feat.to(self.device)
        finestock_feat = finestock_feat.to(self.device)
        
        # 准备输入参数
        input_kwargs = {'market': finemkt_feat, 'device': self.device}
        
        if kwargs['mode'] == 'train':
            # 训练模式
            self.mkt_obs_model.train()
            input_kwargs['deterministic'] = False
            cur_hidden_vector, lambda_val, sigma_val, lambda_log_p, sigma_log_p = self.mkt_obs_model(
                x=finestock_feat, **input_kwargs)

            # 存储结果用于后续训练
            self.cur_hidden_vector_lst.append(cur_hidden_vector)  # (batch, num_of_stocks)
            self.lambda_log_p_lst.append(lambda_log_p)  # (batch,)
            self.sigma_log_p_lst.append(sigma_log_p)  # (batch,)

        elif kwargs['mode'] in ['valid', 'test']:
            # 验证/测试模式
            self.mkt_obs_model.eval()
            with th.no_grad():
                input_kwargs['deterministic'] = True
                cur_hidden_vector, lambda_val, sigma_val, lambda_log_p, sigma_log_p = self.mkt_obs_model(
                    x=finestock_feat, **input_kwargs)
        else:
            raise ValueError(f"Unknown mode: {kwargs['mode']}")

        # 转换为numpy数组
        cur_hidden_vector_ay = cur_hidden_vector.detach().cpu().numpy()  # (batch, num_of_stocks)
        lambda_val_ay = lambda_val.detach().cpu().numpy()  # (batch, )
        sigma_val_ay = sigma_val.detach().cpu().numpy()  # (batch, )
        
        return cur_hidden_vector_ay, lambda_val_ay, sigma_val_ay

    def update_hidden_vec_reward(self, mode, rate_of_price_change, mkt_direction):
        """
        更新隐藏向量奖励。
        
        :param mode: 模式
        :param rate_of_price_change: 价格变化率 (batch, num_of_stocks)
        :param mkt_direction: 市场方向
        """
        self.mkt_obs_model.train()
        rate_of_price_change = th.from_numpy(rate_of_price_change).to(th.float32).to(self.device)
        
        # 获取最后一个隐藏向量
        last_hidden_vec = self.cur_hidden_vector_lst[-1]  # (batch, num_of_stocks)
        
        # 计算当天奖励
        curday_reward = th.log(th.sum((rate_of_price_change-1.0) * last_hidden_vec, dim=-1) + 1.0)  # (batch, )
        
        # 存储奖励和市场方向
        self.hidden_vector_reward_lst.append(curday_reward)
        self.mkt_direction_lst.append(th.from_numpy(mkt_direction).to(self.device))