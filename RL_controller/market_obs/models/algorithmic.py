"""
基于算法的市场观察者模型。
这些模型通常基于传统的金融指标和算法，而不是神经网络。
"""
import numpy as np

from ..registry import register_mkt_obs_model


@register_mkt_obs_model
def ma_1(config, **kwargs):
    """
    创建MA_1模型（移动平均模型）。
    
    :param config: 配置对象
    :param kwargs: 其他参数
    :return: MA_1模型实例
    """
    model = MA_1(config, **kwargs)
    return model


class MA_1:
    """
    基于移动平均的市场观察者模型。
    """
    def __init__(self, config, **kwargs):
        """
        初始化MA_1模型。
        
        :param config: 配置对象
        :param kwargs: 其他参数，必须包含'action_dim'
        """
        super(MA_1, self).__init__()
        self.name = 'ma_1'
        self.config = config
        self.output_action_dim = kwargs['action_dim']  # 传递给RL代理的隐藏向量维度

    def __call__(self, **kwargs):
        """
        调用方法（模拟前向传播）。
        
        :param kwargs: 其他参数，必须包含'stock_cur_close_price'和'stock_ma_price'
        :return: (隐藏向量，lambda值，sigma值)
        """
        # 计算涨跌股票数量
        up_num = np.sum(kwargs['stock_cur_close_price'] > kwargs['stock_ma_price'], axis=1)
        hold_num = np.sum(kwargs['stock_cur_close_price'] == kwargs['stock_ma_price'], axis=1)
        down_num = np.sum(kwargs['stock_cur_close_price'] < kwargs['stock_ma_price'], axis=1)
        
        # 确定市场方向
        direction = np.argmax(np.array([up_num, hold_num, down_num]), axis=0)
        sigma_val_ay = direction
        lambda_val_ay = direction

        # 找出上涨的股票
        up_idx = np.argwhere(np.array(kwargs['stock_cur_close_price'] > kwargs['stock_ma_price']))
        cur_hidden_vector_ay = np.zeros((kwargs['stock_cur_close_price'].shape[0], self.output_action_dim))  # (batch, action_dim)
        
        # 处理没有上涨股票的情况
        zidx = np.argwhere(up_num==0).flatten()
        up_num_norm = np.divide(np.ones_like(up_num), up_num.astype(np.float32), 
                                out=np.zeros_like(up_num)*1.0, where=up_num!=0)

        # 分配权重给上涨股票
        if self.output_action_dim == self.config.topK:
            cur_hidden_vector_ay[up_idx[:,0], up_idx[:, 1]] = up_num_norm[up_idx[:, 0]]
        elif self.output_action_dim == self.config.topK + 1:
            # 第一维是现金
            cur_hidden_vector_ay[up_idx[:,0], up_idx[:, 1]+1] = up_num_norm[up_idx[:, 0]]
        else:
            raise ValueError(f"Unmatch action_dim: {self.output_action_dim}, stock_num: {self.config.topK}")
        
        # 对没有上涨股票的情况均匀分配权重
        cur_hidden_vector_ay[zidx] = 1.0 / self.output_action_dim
        
        return cur_hidden_vector_ay, lambda_val_ay, sigma_val_ay


@register_mkt_obs_model
def dc_1(config, **kwargs):
    """
    创建DC_1模型（方向变化模型）。
    
    :param config: 配置对象
    :param kwargs: 其他参数
    :return: DC_1模型实例
    """
    model = DC_1(config, **kwargs)
    return model


class DC_1:
    """
    基于方向变化的市场观察者模型。
    """
    def __init__(self, config, **kwargs):
        """
        初始化DC_1模型。
        
        :param config: 配置对象
        :param kwargs: 其他参数，必须包含'action_dim'
        """
        super(DC_1, self).__init__()
        self.name = 'dc_1'
        self.config = config
        self.output_action_dim = kwargs['action_dim']  # 传递给RL代理的隐藏向量维度

    def __call__(self, **kwargs):
        """
        调用方法（模拟前向传播）。
        
        :param kwargs: 其他参数，必须包含'dc_events'
        :return: (隐藏向量，lambda值，sigma值)
        """
        # 计算上涨事件数量
        up_events_num = np.sum(kwargs['dc_events'], axis=1)
        fth = kwargs['dc_events'].shape[-1] / 2
        
        # 初始化lambda和sigma值
        lambda_val_ay = np.ones(kwargs['dc_events'].shape[0])
        sigma_val_ay = np.ones(kwargs['dc_events'].shape[0])
        
        # 确定上涨和下跌的样本
        upidx = np.argwhere(up_events_num > fth).flatten()
        downidx = np.argwhere(up_events_num < fth).flatten()
        
        # 设置方向
        lambda_val_ay[upidx] = 0  # 上涨
        lambda_val_ay[downidx] = 2  # 下跌
        sigma_val_ay[upidx] = 0  # 上涨
        sigma_val_ay[downidx] = 2  # 下跌
        
        # 找出上涨事件的股票
        up_idx = np.argwhere(kwargs['dc_events'])
        cur_hidden_vector_ay = np.zeros((kwargs['dc_events'].shape[0], self.output_action_dim))  # (batch, action_dim)
        
        # 处理没有上涨事件的情况
        zidx = np.argwhere(up_events_num==0).flatten()
        up_num_norm = np.divide(np.ones_like(up_events_num), up_events_num.astype(np.float32), 
                                out=np.zeros_like(up_events_num)*1.0, where=up_events_num!=0)
        
        # 分配权重给有上涨事件的股票
        if self.output_action_dim == self.config.topK:
            cur_hidden_vector_ay[up_idx[:,0], up_idx[:, 1]] = up_num_norm[up_idx[:, 0]]
        elif self.output_action_dim == self.config.topK + 1:
            # 第一维是现金
            cur_hidden_vector_ay[up_idx[:,0], up_idx[:, 1]+1] = up_num_norm[up_idx[:, 0]]
        else:
            raise ValueError(f"Unmatch action_dim: {self.output_action_dim}, stock_num: {self.config.topK}")
        
        # 对没有上涨事件的情况均匀分配权重
        cur_hidden_vector_ay[zidx] = 1.0 / self.output_action_dim
        
        return cur_hidden_vector_ay, lambda_val_ay, sigma_val_ay