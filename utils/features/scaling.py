#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         feature_scaling.py
 Description:  Feature scaling and dataset splitting functions
 Author:       MASA
---------------------------------
'''
import numpy as np
import pandas as pd
import copy

def scale_and_split_features(data, raw_col_lst, tech_indicator_lst, config):
    """
    Scale features and split data into train/valid/test sets.
    
    Args:
        data: DataFrame with generated features
        raw_col_lst: List of original columns
        tech_indicator_lst: List of technical indicators
        config: Configuration object
        
    Returns:
        Dictionary with dataset splits
    """
    data['date'] = pd.to_datetime(data['date'])
    datax = copy.deepcopy(data)

    # Add covariance calculation if enabled
    if config.enable_cov_features:
        datax = add_covariance_features(datax, config)

    # Add daily returns if configured
    if 'DAILYRETURNS-{}'.format(config.dailyRetun_lookback) in config.otherRef_indicator_lst:
        datax = add_daily_returns(datax, config)

    datax.reset_index(drop=True, inplace=True)
    if config.test_date_end is None:
        if config.valid_date_end is None:
            data_date_end = config.train_date_end
        else:
            data_date_end = config.valid_date_end
    else:
        data_date_end = config.test_date_end

    # Process data before training period
    data_bftrain = copy.deepcopy(datax[datax['date'] < config.train_date_start][['date', 'stock', 'DAILYRETURNS-{}'.format(config.dailyRetun_lookback)]])
    data_bftrain = data_bftrain.dropna(axis=0, how='any')

    # Filter data for relevant date range
    datax = copy.deepcopy(datax[(datax['date'] >= config.train_date_start) & (datax['date'] <= data_date_end)])
    datax.sort_values(['date', 'stock'], ascending=True, inplace=True, ignore_index=True) 
    
    # Normalize technical indicators
    datax = normalize_indicators(datax, tech_indicator_lst, config)
    
    if config.enable_cov_features:
        tech_indicator_lst = list(tech_indicator_lst) + ['cov']
    cols_order = list(raw_col_lst) + list(config.otherRef_indicator_lst) + list(sorted(tech_indicator_lst)) 
    datax = datax[cols_order]
    
    # Split data into train/valid/test sets
    dataset_dict = create_dataset_splits(datax, data_bftrain, config)
    
    print(datax)
    return dataset_dict

def add_covariance_features(datax, config):
    """Add covariance features to the dataset"""
    datax.sort_values(['date', 'stock'], ascending=True, inplace=True, ignore_index=True)
    datax.index = datax.date.factorize()[0]
    cov_lst = []
    date_lst = []
    for idx in range(config.cov_lookback, datax['date'].nunique()):
        sigPeriodData = datax.loc[idx-config.cov_lookback:idx, :]
        sigPeriodClose = sigPeriodData.pivot_table(index = 'date',columns = 'stock', values = 'close')
        sigPeriodClose.sort_values(['date'], ascending=True, inplace=True)
        sigPeriodReturn = sigPeriodClose.pct_change().dropna()
        covs = sigPeriodReturn.cov().values 
        cov_lst.append(covs)
        date_lst.append(datax.loc[idx, 'date'].values[0])
    
    cov_pd = pd.DataFrame({'date': date_lst, 'cov': cov_lst})
    return pd.merge(datax, cov_pd, how='inner', on=['date'])

def add_daily_returns(datax, config):
    """Add daily returns to the dataset"""
    r_lst = []
    stockNo_lst = []
    date_lst = []
    datax.sort_values(['date', 'stock'], ascending=True, inplace=True)
    datax.reset_index(drop=True, inplace=True)
    datax.index = datax.date.factorize()[0]    
    for idx in range(config.dailyRetun_lookback, datax['date'].nunique()):
        sigPeriodData = datax.loc[idx-config.dailyRetun_lookback:idx, :][['date', 'stock', 'close']]
        sigPeriodClose = sigPeriodData.pivot_table(index = 'date',columns = 'stock', values = 'close')
        sigPeriodClose.sort_values(['date'], ascending=True, inplace=True)
        sigPeriodReturn = sigPeriodClose.pct_change().dropna() # without percentage
        sigPeriodReturn.sort_values(['date'], ascending=True, inplace=True)
        sigStockName_lst = np.array(sigPeriodReturn.columns)
        stockNo_lst = stockNo_lst + list(sigStockName_lst)
        r_lst = r_lst + list(np.transpose(sigPeriodReturn.values))
        date_lst = date_lst + [datax.loc[idx, 'date'].values[0]] * len(sigStockName_lst)
    r_pd = pd.DataFrame({'date': date_lst, 'stock': stockNo_lst, 'DAILYRETURNS-{}'.format(config.dailyRetun_lookback): r_lst})
    return pd.merge(datax, r_pd, how='inner', on=['date', 'stock'])

def normalize_indicators(datax, tech_indicator_lst, config):
    """Normalize technical indicators based on training data"""
    for sigIndicatorName in tech_indicator_lst:
        # Feature normalization
        nan_cnt = len(np.argwhere(np.isnan(np.array(datax[sigIndicatorName]))))
        inf_cnt = len(np.argwhere(np.isinf(np.array(datax[sigIndicatorName]))))
        if (nan_cnt > 0) or (inf_cnt > 0):
            raise ValueError("Indicator: {}, nan count: {}, inf count: {}".format(sigIndicatorName, nan_cnt, inf_cnt))
        if (sigIndicatorName in ['CHANGELOGCLOSE', 'cov']) or ('close_w' in sigIndicatorName) or ('open_w' in sigIndicatorName) or ('high_w' in sigIndicatorName) or ('low_w' in sigIndicatorName) or ('volume_w' in sigIndicatorName):
            # No need to be normalized.
            continue
        train_ay = np.array(datax[(datax['date'] >= config.train_date_start) & (datax['date'] <= config.train_date_end)][sigIndicatorName])
        ind_mean = np.mean(train_ay)
        ind_std = np.std(train_ay, ddof=1)
        datax[sigIndicatorName] = (np.array(datax[sigIndicatorName]) - ind_mean) / ind_std
    return datax

def create_dataset_splits(datax, data_bftrain, config):
    """Create train/valid/test dataset splits"""
    dataset_dict = {}
    train_data = copy.deepcopy(datax[(datax['date'] >= config.train_date_start) & (datax['date'] <= config.train_date_end)])
    train_data.sort_values(['date', 'stock'], ascending=True, inplace=True, ignore_index=True)
    dataset_dict['train'] = train_data
    
    if (config.valid_date_start is not None) and (config.valid_date_end is not None):
        valid_data = copy.deepcopy(datax[(datax['date'] >= config.valid_date_start) & (datax['date'] <= config.valid_date_end)])
        valid_data.sort_values(['date', 'stock'], ascending=True, inplace=True, ignore_index=True)
        dataset_dict['valid'] = valid_data

    if (config.test_date_start is not None) and (config.test_date_end is not None):
        test_data = copy.deepcopy(datax[(datax['date'] >= config.test_date_start) & (datax['date'] <= config.test_date_end)])
        test_data.sort_values(['date', 'stock'], ascending=True, inplace=True, ignore_index=True)
        dataset_dict['test'] = test_data

    data_bftrain.sort_values(['date', 'stock'], ascending=True, inplace=True, ignore_index=True)
    dataset_dict['bftrain'] = data_bftrain
    
    return dataset_dict