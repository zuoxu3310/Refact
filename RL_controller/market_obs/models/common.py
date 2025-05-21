"""
市场观察者模型的通用组件。
"""
import torch as th
import torch.nn as nn


class GenScore(nn.Module):
    """
    分数生成器模块。
    生成并返回分数值和相应的对数概率。
    """
    def __init__(self, **kwargs):
        """初始化GenScore模块"""
        super(GenScore, self).__init__()
        self.name = 'gen_score'
        # self.sigmoid_fn = nn.Sigmoid()
    
    def forward(self, x, **kwargs):
        """
        前向传播方法。
        
        :param x: 输入张量
        :param kwargs: 其他参数
        :return: 分数和分数的对数概率
        """
        score_log_p = x
        score = th.argmax(x.detach(), dim=1)
        return score, score_log_p