"""
TD3控制器实现的工具函数。
"""
from typing import List, Type
import torch as th


def create_mlp_adj(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[th.nn.Module] = th.nn.ReLU,
    squash_output: bool = False,
) -> List[th.nn.Module]:
    """
    创建多层感知器(MLP)，即一系列每层后跟激活函数的全连接层。
    
    :param input_dim: 输入向量的维度
    :param output_dim: 输出维度
    :param net_arch: 神经网络架构
        表示每层的单元数。
        此列表的长度即为层数。
    :param activation_fn: 每层后使用的激活函数
    :param squash_output: 是否使用Softmax函数压缩输出
    :return: 神经网络模块列表
    """
    if len(net_arch) > 0:
        modules = [th.nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(th.nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(th.nn.Linear(last_layer_dim, output_dim))
        
    if squash_output:
        # modules.append(th.nn.Tanh())
        modules.append(th.nn.Softmax(dim=1))
        
    return modules