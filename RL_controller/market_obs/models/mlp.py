"""
基于多层感知器(MLP)的市场观察者模型。
"""
import torch as th
import torch.nn as nn

from ..registry import register_mkt_obs_model
from .common import GenScore


@register_mkt_obs_model
def mlp_1(config, **kwargs):
    """
    创建MLP_1模型。
    
    :param config: 配置对象
    :param kwargs: 其他参数
    :return: MLP_1模型实例
    """
    model = MLP_1(config, **kwargs)
    return model


class MLP_1(nn.Module):
    """
    基于MLP的市场观察者模型。
    """
    def __init__(self, config, **kwargs):
        """
        初始化MLP_1模型。
        
        :param config: 配置对象
        :param kwargs: 其他参数，必须包含'action_dim'
        """
        super(MLP_1, self).__init__()
        self.name = 'mlp_1'
        self.config = config
        self.output_action_dim = kwargs['action_dim']  # 传递给RL代理的隐藏向量维度

        # 股票处理部分
        feat_s_length = self.config.topK * len(self.config.use_features) * self.config.fine_window_size
        self.flatten_s = nn.Flatten() 
        self.fc1_s = nn.Linear(feat_s_length, 256, bias=True)
        self.relu1_s = nn.ReLU()
        self.fc2_s = nn.Linear(256, 64, bias=True)
        self.shortcut_s = nn.Linear(feat_s_length, 64, bias=True)
        self.relu2_s = nn.ReLU()
        
        # 市场处理部分
        feat_m_length = len(self.config.use_features) * self.config.fine_window_size
        self.flatten_m = nn.Flatten()
        self.fc1_m = nn.Linear(feat_m_length, 16, bias=True)
        self.relu1_m = nn.ReLU()
        self.fc2_m = nn.Linear(16, 16, bias=True)
        self.relu2_m = nn.ReLU()

        # 合并和输出
        self.fc_merge = nn.Linear(80, self.output_action_dim, bias=True) 
        self.sm = nn.Softmax(dim=1)

        # lambda和sigma输出
        self.fc_lambda = nn.Linear(80, 3, bias=True)
        self.gen_lambda = GenScore()

        self.fc_sigma = nn.Linear(80, 3, bias=True)
        self.gen_sigma = GenScore()

    def forward(self, x, **kwargs):
        """
        前向传播方法。
        
        :param x: 股票特征，形状为(batch, features, num_of_stocks, window_size)
        :param kwargs: 其他参数，必须包含'market'、'deterministic'和'device'
        :return: (隐藏向量，lambda值，sigma值，lambda对数概率，sigma对数概率)
        """
        # 股票处理
        s0 = self.flatten_s(x)
        s1 = self.fc1_s(s0)
        s1 = self.relu1_s(s1)
        s1 = self.fc2_s(s1)
        s2 = self.shortcut_s(s0)  # 快捷连接
        s3 = s1 + s2  # 相加
        s3 = self.relu2_s(s3)

        # 市场处理
        m = kwargs['market']
        m0 = self.flatten_m(m)
        m1 = self.fc1_m(m0)
        m1 = self.relu1_m(m1)
        m1 = self.fc2_m(m1)
        m2 = m1 + m0
        m2 = self.relu2_m(m2)

        # 合并
        k1 = th.cat((s3, m2), dim=1)
        k2 = self.fc_merge(k1)  # 输出形状取决于是否考虑现金: (batch, num_of_stocks) or (batch, num_of_stocks+1)
        hidden_vec = self.sm(k2)  # 隐藏向量

        # 生成lambda和sigma值
        lambda_vec = self.fc_lambda(k1)
        lambda_kwargs = {
            'name': 'lambda', 
            'deterministic': kwargs['deterministic'], 
            'device': kwargs['device'], 
            'score_min': self.config.lambda_min, 
            'score_max': self.config.lambda_max
        }
        lambda_val, lambda_log_p = self.gen_lambda(lambda_vec, **lambda_kwargs)
        
        sigma_vec = self.fc_sigma(k1)
        sigma_kwargs = {
            'name': 'sigma', 
            'deterministic': kwargs['deterministic'], 
            'device': kwargs['device'], 
            'score_min': self.config.sigma_min, 
            'score_max': self.config.sigma_max
        }
        sigma_val, sigma_log_p = self.gen_sigma(sigma_vec, **sigma_kwargs)

        return hidden_vec, lambda_val, sigma_val, lambda_log_p, sigma_log_p


@register_mkt_obs_model
def stf_1(config, **kwargs):
    """
    创建STF_1模型（占位实现）。
    
    :param config: 配置对象
    :param kwargs: 其他参数
    :return: STF_1模型实例
    """
    model = STF_1(config, **kwargs)
    return model


class STF_1(nn.Module):
    """
    STF类型的市场观察者模型（占位实现）。
    """
    def __init__(self, config, **kwargs):
        """
        初始化STF_1模型。
        
        :param config: 配置对象
        :param kwargs: 其他参数
        """
        super(STF_1, self).__init__()
        self.name = 'stf_1'
        self.config = config
        
    def forward(self, x):
        """
        前向传播方法（占位实现）。
        
        :param x: 输入
        :return: 输入（原样返回）
        """
        # 占位实现
        return x