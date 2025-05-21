"""
基于LSTM的市场观察者模型。
"""
import torch as th
import torch.nn as nn

from ..registry import register_mkt_obs_model
from .common import GenScore


@register_mkt_obs_model
def lstm_1(config, **kwargs):
    """
    创建LSTM_1模型。
    
    :param config: 配置对象
    :param kwargs: 其他参数
    :return: LSTM_1模型实例
    """
    model = LSTM_1(config, **kwargs)
    return model


class LSTM_1(nn.Module):
    """
    基于LSTM的市场观察者模型。
    """
    def __init__(self, config, **kwargs):
        """
        初始化LSTM_1模型。
        
        :param config: 配置对象
        :param kwargs: 其他参数，必须包含'action_dim'
        """
        super(LSTM_1, self).__init__()
        self.name = 'lstm_1'
        self.config = config
        self.output_action_dim = kwargs['action_dim']  # 传递给RL代理的隐藏向量维度

        self.in_features = len(self.config.use_features) * (self.config.topK + 1)  # +1用于市场数据
        self.window_len = self.config.fine_window_size
        hidden_dim = 128

        self.flatten_s = nn.Flatten(start_dim=1, end_dim=2)

        self.lstm = nn.LSTM(input_size=self.in_features, hidden_size=hidden_dim)
        self.attn1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn2 = nn.Linear(hidden_dim, 1)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc_merge = nn.Linear(hidden_dim, self.output_action_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

        # lambda和sigma输出
        self.fc_lambda = nn.Linear(hidden_dim, 3, bias=True)
        self.gen_lambda = GenScore()

        self.fc_sigma = nn.Linear(hidden_dim, 3, bias=True)
        self.gen_sigma = GenScore()

    def forward(self, x, **kwargs):
        """
        前向传播方法。
        
        :param x: 股票特征，形状为(batch, features, num_of_stocks, window_size)
        :param kwargs: 其他参数，必须包含'market'、'deterministic'和'device'
        :return: (隐藏向量，lambda值，sigma值，lambda对数概率，sigma对数概率)
        """
        # 处理股票数据
        # (batch, features, num_of_stocks, window_size) -> (batch, features*num_of_stocks, window_size)
        s0 = self.flatten_s(x) 
        m = kwargs['market']
        x1 = th.cat((s0, m), dim=1)  # -> (batch, mkt_feat+stock_feat, window_size)
        x1 = x1.permute(2, 0, 1)  # -> (window_size, batch, mkt_feat+stock_feat)
        
        # LSTM处理
        outputs, (h_n, c_n) = self.lstm(x1)  # outputs: (window_size, batch, hidden_size), hn:(1, batch, hidden_size)
        
        # 注意力机制
        H_n = h_n.repeat((self.window_len, 1, 1))  # (window_len, batch, hidden_size)
        scores = self.attn2(th.tanh(self.attn1(th.cat([outputs, H_n], dim=2))))  # (win_len, batch, 1)
        scores = scores.squeeze(2).transpose(1, 0)  # (batch, win_len)
        attn_weights = th.softmax(scores, dim=1)  # (batch, win_len)

        outputs = outputs.permute(1, 0, 2)  # (batch, win_len, hidden_size)
        attn_embed = th.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)  # (batch, hidden_size)
        
        # 批量归一化处理
        if attn_embed.size(0) == 1:
            embed = th.relu(self.linear1(attn_embed))  # (batch, hidden_size)
        else:
            embed = th.relu(self.bn1(self.linear1(attn_embed)))  # (batch, hidden_size)
        
        # 生成隐藏向量
        hidden_vec = self.fc_merge(embed)  # 输出形状取决于是否考虑现金
        hidden_vec = self.sm(hidden_vec)  # 使用softmax归一化
        
        # 生成lambda和sigma值
        lambda_vec = self.fc_lambda(embed)
        lambda_kwargs = {
            'name': 'lambda', 
            'deterministic': kwargs['deterministic'], 
            'device': kwargs['device'], 
            'score_min': self.config.lambda_min, 
            'score_max': self.config.lambda_max
        }
        lambda_val, lambda_log_p = self.gen_lambda(lambda_vec, **lambda_kwargs)
        
        sigma_vec = self.fc_sigma(embed)
        sigma_kwargs = {
            'name': 'sigma', 
            'deterministic': kwargs['deterministic'], 
            'device': kwargs['device'], 
            'score_min': self.config.sigma_min, 
            'score_max': self.config.sigma_max
        }
        sigma_val, sigma_log_p = self.gen_sigma(sigma_vec, **sigma_kwargs)

        return hidden_vec, lambda_val, sigma_val, lambda_log_p, sigma_log_p