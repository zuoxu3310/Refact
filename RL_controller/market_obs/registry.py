"""
市场观察者模型的注册机制。
"""
from typing import Dict, Callable, Any

# 存储所有注册的市场观察者模型的字典
_mkt_obs_model_entrypoints: Dict[str, Callable[..., Any]] = {}

def register_mkt_obs_model(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    注册一个市场观察者模型。
    用作装饰器。
    
    :param fn: 创建模型的函数
    :return: 原函数不变
    """
    net_name = fn.__name__  # 例如 'deeptradernet_v1'
    _mkt_obs_model_entrypoints[net_name] = fn
    return fn

def is_model(net_name: str) -> bool:
    """
    检查指定名称的模型是否已注册。
    
    :param net_name: 模型名称
    :return: 如果模型已注册返回True，否则返回False
    """
    return net_name in _mkt_obs_model_entrypoints

def mkt_obs_model_entrypoint(net_name: str) -> Callable[..., Any]:
    """
    获取指定名称模型的入口点函数。
    
    :param net_name: 模型名称
    :return: 创建模型的函数
    """
    return _mkt_obs_model_entrypoints[net_name]

def create_mkt_obs_model(config, **kwargs):
    """
    创建市场观察者模型。
    
    模型网络名称格式: 'mlp_1', {arch_name}-{rc_version}
    
    :param config: 配置对象
    :param kwargs: 传递给模型创建函数的其他参数
    :return: 创建的模型
    :raises ValueError: 如果模型架构未知
    """
    if not is_model(net_name=config.mktobs_algo):
        algo_arch, algo_verison = config.mktobs_algo.split('_', 1)
        raise ValueError(f"Unknown market observer model/architecture: {algo_arch}, tag: {algo_verison}")
    
    create_fn = mkt_obs_model_entrypoint(net_name=config.mktobs_algo)
    model = create_fn(
        config=config,
        **kwargs,
    )
    # Load checkpoint if needed
    return model