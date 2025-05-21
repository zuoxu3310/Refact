"""
控制器模块的通用工具函数。
"""
import numpy as np
import scipy.stats as spstats


def calculate_portfolio_risk(weights, covariance_matrix):
    """
    计算投资组合的风险（标准差）。
    
    :param weights: 投资组合权重
    :param covariance_matrix: 协方差矩阵
    :return: 投资组合风险（标准差）
    """
    return np.sqrt(np.matmul(np.matmul(weights, covariance_matrix), weights.T))


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    计算夏普比率。
    
    :param returns: 收益率序列
    :param risk_free_rate: 无风险利率
    :return: 夏普比率
    """
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0


def calculate_var(returns, confidence_level=0.95):
    """
    计算风险价值(VaR)。
    
    :param returns: 收益率序列
    :param confidence_level: 置信水平
    :return: VaR值
    """
    return -1 * np.percentile(returns, 100 * (1 - confidence_level))


def calculate_cvar(returns, confidence_level=0.95):
    """
    计算条件风险价值(CVaR)。
    
    :param returns: 收益率序列
    :param confidence_level: 置信水平
    :return: CVaR值
    """
    var = calculate_var(returns, confidence_level)
    return -1 * np.mean(returns[returns <= -var])


def normalize_weights(weights, lower_bound=0.0, upper_bound=1.0):
    """
    归一化权重到指定范围。
    
    :param weights: 原始权重
    :param lower_bound: 权重下限
    :param upper_bound: 权重上限
    :return: 归一化后的权重
    """
    weights = np.array(weights)
    if np.sum(np.abs(weights)) > 0:
        return np.clip(weights / np.sum(np.abs(weights)), lower_bound, upper_bound)
    return weights