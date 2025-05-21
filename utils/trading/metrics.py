#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         metrics.py
 Description:  Performance metrics calculation for portfolio.
 Author:       MASA
---------------------------------
'''
import numpy as np
import scipy.stats as spstats

def calculate_portfolio_metrics(env):
    """
    Calculate portfolio performance metrics.
    
    Args:
        env: Trading environment instance with history data
        
    Returns:
        Dictionary of calculated metrics
    """
    profit_lst = np.array(env.profit_lst)
    asset_lst = np.array(env.asset_lst)

    netProfit = env.cur_capital - env.initial_asset # Profits
    netProfit_pct = netProfit / env.initial_asset # Rate of overall returns

    diffPeriodAsset = np.diff(asset_lst)
    sigReturn_max = np.max(diffPeriodAsset) # Maximal returns in a single transaction.
    sigReturn_min = np.min(diffPeriodAsset) # Minimal returns in a single transaction

    # Annual Returns
    annualReturn_pct = np.power((1 + netProfit_pct), (env.config.tradeDays_per_year/len(asset_lst))) - 1

    dailyReturn_pct_max = np.max(profit_lst)
    dailyReturn_pct_min = np.min(profit_lst)
    avg_dailyReturn_pct = np.mean(profit_lst)
    
    # Strategy volatility
    volatility = np.sqrt(np.sum(np.power((profit_lst - avg_dailyReturn_pct), 2)) * env.config.tradeDays_per_year / (len(profit_lst) - 1))

    # SR_Vol, Long-term risk
    sharpeRatio = ((annualReturn_pct * 100) - env.config.mkt_rf[env.config.market_name])/ (volatility * 100)

    # Daily Sharpe Ratio
    dailyAnnualReturn_lst = np.power((1+np.array(profit_lst)), env.config.tradeDays_per_year) - 1
    dailyRisk_lst = np.array(env.risk_cbf_lst) * np.sqrt(env.config.tradeDays_per_year) # Daily Risk to Annual Risk
    dailySR = ((dailyAnnualReturn_lst[1:] * 100) - env.config.mkt_rf[env.config.market_name]) / (dailyRisk_lst[1:] * 100)
    dailySR = np.append(0, dailySR)
    dailySR_max = np.max(dailySR)
    dailySR_min = np.min(dailySR[dailySR!=0])
    dailySR_avg = np.mean(dailySR)

    # For performance analysis without CBF
    dailyReturnRate_wocbf = np.diff(env.return_raw_lst) / np.array(env.return_raw_lst)[:-1]
    dailyReturnRate_wocbf = np.append(0, dailyReturnRate_wocbf)
    dailyAnnualReturn_wocbf_lst = np.power((1+dailyReturnRate_wocbf), env.config.tradeDays_per_year) - 1
    dailyRisk_wocbf_lst = np.array(env.risk_raw_lst) * np.sqrt(env.config.tradeDays_per_year)  
    dailySR_wocbf = ((dailyAnnualReturn_wocbf_lst[1:] * 100) - env.config.mkt_rf[env.config.market_name]) / (dailyRisk_wocbf_lst[1:] * 100)
    dailySR_wocbf = np.append(0, dailySR_wocbf)
    dailySR_wocbf_max = np.max(dailySR_wocbf)
    dailySR_wocbf_min = np.min(dailySR_wocbf[dailySR_wocbf!=0])
    dailySR_wocbf_avg = np.mean(dailySR_wocbf)

    # Metrics without CBF
    annualReturn_wocbf_pct = np.power((1 + ((env.return_raw_lst[-1] - env.initial_asset) / env.initial_asset)), (env.config.tradeDays_per_year/len(env.return_raw_lst))) - 1
    volatility_wocbf = np.sqrt((np.sum(np.power((dailyReturnRate_wocbf - np.mean(dailyReturnRate_wocbf)), 2)) * env.config.tradeDays_per_year / (len(env.return_raw_lst) - 1)))
    sharpeRatio_woCBF = ((annualReturn_wocbf_pct * 100) - env.config.mkt_rf[env.config.market_name])/ (volatility_wocbf * 100)
    sharpeRatio_woCBF = np.max([sharpeRatio_woCBF, 0])

    winRate = len(np.argwhere(diffPeriodAsset>0))/(len(diffPeriodAsset) + 1)

    # Maximum Drawdown (MDD)
    repeat_asset_lst = np.tile(asset_lst, (len(asset_lst), 1))
    mdd_mtix = np.triu(1 - repeat_asset_lst / np.reshape(asset_lst, (-1, 1)), k=1)
    mddmaxidx = np.argmax(mdd_mtix)
    mdd_highidx = mddmaxidx // len(asset_lst)
    mdd_lowidx = mddmaxidx % len(asset_lst)
    mdd = np.max(mdd_mtix)
    mdd_high = asset_lst[mdd_highidx]
    mdd_low = asset_lst[mdd_lowidx]
    mdd_highTimepoint = env.date_memory[mdd_highidx]
    mdd_lowTimepoint = env.date_memory[mdd_lowidx]

    # Strategy volatility during trading
    cumsum_r = np.cumsum(profit_lst)/np.arange(1, env.totalTradeDay+1) # average cumulative returns rate
    repeat_profit_lst = np.tile(profit_lst, (len(profit_lst), 1))
    stg_vol_lst = np.sqrt(np.sum(np.power(np.tril(repeat_profit_lst - np.reshape(cumsum_r, (-1,1)), k=0), 2), axis=1)[1:] / np.arange(1, len(repeat_profit_lst)) * env.config.tradeDays_per_year)
    stg_vol_lst = np.append([0], stg_vol_lst, axis=0)

    vol_max = np.max(stg_vol_lst)
    vol_min = np.min(np.array(stg_vol_lst)[np.array(stg_vol_lst)!=0])
    vol_avg = np.mean(stg_vol_lst)

    # Short-term risk metrics
    risk_max = np.max(env.risk_cbf_lst)
    risk_min = np.min(np.array(env.risk_cbf_lst)[np.array(env.risk_cbf_lst)!=0])
    risk_avg = np.mean(env.risk_cbf_lst)

    risk_raw_max = np.max(env.risk_raw_lst)
    risk_raw_min = np.min(np.array(env.risk_raw_lst)[np.array(env.risk_raw_lst)!=0])
    risk_raw_avg = np.mean(env.risk_raw_lst)

    # Downside risk at volatility        
    risk_downsideAtVol_daily = np.sqrt(np.sum(np.power(np.tril((repeat_profit_lst - np.reshape(cumsum_r, (-1,1))) * (repeat_profit_lst<np.reshape(cumsum_r, (-1,1))), k=0), 2), axis=1)[1:] / np.arange(1, len(repeat_profit_lst)) * env.config.tradeDays_per_year)
    risk_downsideAtVol_daily = np.append([0], risk_downsideAtVol_daily, axis=0)
    risk_downsideAtVol = risk_downsideAtVol_daily[-1]
    risk_downsideAtVol_daily_max = np.max(risk_downsideAtVol_daily)
    risk_downsideAtVol_daily_min = np.min(risk_downsideAtVol_daily)
    risk_downsideAtVol_daily_avg = np.mean(risk_downsideAtVol_daily)

    # Downside risk at value against initial capital
    risk_downsideAtValue_daily = (asset_lst / env.initial_asset) - 1
    risk_downsideAtValue_daily_max = np.max(risk_downsideAtValue_daily)
    risk_downsideAtValue_daily_min = np.min(risk_downsideAtValue_daily)
    risk_downsideAtValue_daily_avg = np.mean(risk_downsideAtValue_daily)

    # CVaR metrics
    cvar_max = np.max(env.cvar_lst)
    cvar_min = np.min(np.array(env.cvar_lst)[np.array(env.cvar_lst)!=0])
    cvar_avg = np.mean(env.cvar_lst)

    cvar_raw_max = np.max(env.cvar_raw_lst)
    cvar_raw_min = np.min(np.array(env.cvar_raw_lst)[np.array(env.cvar_raw_lst)!=0])
    cvar_raw_avg = np.mean(env.cvar_raw_lst)

    # Calmar ratio
    time_T = len(profit_lst)
    avg_return = netProfit_pct / time_T
    variance_r = np.sum(np.power((profit_lst - avg_dailyReturn_pct), 2)) / (len(profit_lst) - 1)
    volatility_daily = np.sqrt(variance_r)
 
    if netProfit_pct > 0:
        shrp = avg_return / volatility_daily
        calmarRatio = (time_T * np.power(shrp, 2)) / (0.63519 + 0.5 * np.log(time_T) + np.log(shrp))
    elif netProfit_pct == 0:
        calmarRatio = (netProfit_pct) / (1.2533 * volatility_daily * np.sqrt(time_T))
    else:
        # netProfit_pct < 0
        calmarRatio = (netProfit_pct) / (-(avg_return * time_T) - (variance_r / avg_return))

    # Sterling ratio
    move_mdd_mask = np.where(np.array(profit_lst)<0, 1, 0)
    moving_mdd = np.sqrt(np.sum(np.power(profit_lst * move_mdd_mask, 2)) * env.config.tradeDays_per_year / (len(profit_lst) - 1))
    sterlingRatio = ((annualReturn_pct * 100) - env.config.mkt_rf[env.config.market_name]) / (moving_mdd * 100)

    # CBF contribution
    cbf_abssum_contribution = np.sum(np.abs(env.action_cbf_memeory[:-1]))

    # Create results dictionary
    info_dict = {
        'ep': env.epoch, 
        'trading_days': env.totalTradeDay, 
        'annualReturn_pct': annualReturn_pct, 
        'volatility': volatility, 
        'sharpeRatio': sharpeRatio, 
        'sharpeRatio_wocbf': sharpeRatio_woCBF,
        'mdd': mdd, 
        'calmarRatio': calmarRatio, 
        'sterlingRatio': sterlingRatio, 
        'netProfit': netProfit, 
        'netProfit_pct': netProfit_pct, 
        'winRate': winRate,
        'vol_max': vol_max, 
        'vol_min': vol_min, 
        'vol_avg': vol_avg,
        'risk_max': risk_max, 
        'risk_min': risk_min, 
        'risk_avg': risk_avg,
        'riskRaw_max': risk_raw_max, 
        'riskRaw_min': risk_raw_min, 
        'riskRaw_avg': risk_raw_avg,
        'dailySR_max': dailySR_max, 
        'dailySR_min': dailySR_min, 
        'dailySR_avg': dailySR_avg, 
        'dailySR_wocbf_max': dailySR_wocbf_max, 
        'dailySR_wocbf_min': dailySR_wocbf_min, 
        'dailySR_wocbf_avg': dailySR_wocbf_avg,
        'dailyReturn_pct_max': dailyReturn_pct_max, 
        'dailyReturn_pct_min': dailyReturn_pct_min, 
        'dailyReturn_pct_avg': avg_dailyReturn_pct,
        'sigReturn_max': sigReturn_max, 
        'sigReturn_min': sigReturn_min, 
        'mdd_high': mdd_high, 
        'mdd_low': mdd_low, 
        'mdd_high_date': mdd_highTimepoint, 
        'mdd_low_date': mdd_lowTimepoint, 
        'final_capital': env.cur_capital, 
        'reward_sum': np.sum(env.reward_lst),
        'final_capital_wocbf': env.return_raw_lst[-1], 
        'cbf_contribution': cbf_abssum_contribution,
        'risk_downsideAtVol': risk_downsideAtVol, 
        'risk_downsideAtVol_daily_max': risk_downsideAtVol_daily_max, 
        'risk_downsideAtVol_daily_min': risk_downsideAtVol_daily_min, 
        'risk_downsideAtVol_daily_avg': risk_downsideAtVol_daily_avg,
        'risk_downsideAtValue_daily_max': risk_downsideAtValue_daily_max, 
        'risk_downsideAtValue_daily_min': risk_downsideAtValue_daily_min, 
        'risk_downsideAtValue_daily_avg': risk_downsideAtValue_daily_avg,
        'cvar_max': cvar_max, 
        'cvar_min': cvar_min, 
        'cvar_avg': cvar_avg, 
        'cvar_raw_max': cvar_raw_max, 
        'cvar_raw_min': cvar_raw_min, 
        'cvar_raw_avg': cvar_raw_avg,
        'solver_solvable': env.solver_stat['solvable'], 
        'solver_insolvable': env.solver_stat['insolvable'], 
        'cputime': env.end_cputime - env.start_cputime - (env.exclusive_cputime if env.mode == 'train' else 0), 
        'systime': env.end_systime - env.start_systime - (env.exclusive_systime if env.mode == 'train' else 0),
        'asset_lst': np.copy(asset_lst), 
        'daily_return_lst': np.copy(profit_lst), 
        'reward_lst': np.copy(env.reward_lst), 
        'stg_vol_lst': np.copy(stg_vol_lst), 
        'risk_lst': np.copy(env.risk_cbf_lst), 
        'risk_wocbf_lst': np.copy(env.risk_raw_lst),
        'capital_wocbf_lst': np.copy(env.return_raw_lst), 
        'daily_sr_lst': np.copy(dailySR), 
        'daily_sr_wocbf_lst': np.copy(dailySR_wocbf),
        'risk_adj_lst': np.copy(env.risk_adj_lst), 
        'ctrl_weight_lst': np.copy(env.ctrl_weight_lst), 
        'solvable_flag': np.copy(env.solvable_flag), 
        'risk_pred_lst': np.copy(env.risk_pred_lst),
        'final_action_abssum_lst': np.copy(np.sum(np.abs(np.array(env.actions_memory)), axis=1)), 
        'rl_action_abssum_lst': np.copy(np.sum(np.abs(np.array(env.action_rl_memory)), axis=1)[:-1]), 
        'cbf_action_abssum_lst': np.copy(np.sum(np.abs(np.array(env.action_cbf_memeory)), axis=1)[:-1]), 
        'daily_downsideAtVol_risk_lst': np.copy(risk_downsideAtVol_daily), 
        'daily_downsideAtValue_risk_lst': np.copy(risk_downsideAtValue_daily),
        'cvar_lst': np.copy(env.cvar_lst), 
        'cvar_raw_lst': np.copy(env.cvar_raw_lst),
    }

    return info_dict, mdd, mdd_highTimepoint, mdd_lowTimepoint, mdd_high, mdd_low