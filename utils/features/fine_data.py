#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         fine_data_processor.py
 Description:  Process fine-grained market and stock data
 Author:       MASA
---------------------------------
'''
import numpy as np
import pandas as pd
import copy
import os
from talib import abstract
# Update this import
from utils.features.directional_change import dc_feature_generation

def process_fine_data(data, config):
    """
    Process fine-grained market and stock data.
    
    Args:
        data: Dictionary with dataset splits
        config: Configuration object
        
    Returns:
        Dictionary with dataset splits and additional fine data
    """
    # Preprocess the data for the market observer.
    # Fine market data
    fine_mkt_data = generate_market_features(config, freq=config.finefreq)
    # Fine stock data
    fine_stock_data = generate_fine_stock_features(config)

    # Add fine data to train/valid/test sets
    data = add_fine_data_to_datasets(data, fine_mkt_data, fine_stock_data, config)
    
    return data

def add_fine_data_to_datasets(data, fine_mkt_data, fine_stock_data, config):
    """Add fine data to each dataset split"""
    # Train
    daily_date_lst = data['train']['date'].unique()
    extra_train_data = {}

    fmd_train = copy.deepcopy(fine_mkt_data[(fine_mkt_data['date'] >= config.train_date_start) & (fine_mkt_data['date'] <= config.train_date_end)])
    fmd_train.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)
    extra_train_data['fine_market'] = fmd_train
    fmd_date_lst = fmd_train['date'].unique()
    
    # Validate train fine market dates match daily dates
    if (len(set(fmd_date_lst) - set(daily_date_lst)) != 0) or (len(set(daily_date_lst) - set(fmd_date_lst)) != 0):
        raise ValueError("[Train, fine market] | ref date number: {}, extract date number: {} \nmissing dates in ref: {}\nmissing dates in extraction: {}".format(
            len(fmd_date_lst), len(daily_date_lst), set(fmd_date_lst) - set(daily_date_lst), set(daily_date_lst) - set(fmd_date_lst)))

    fsd_train = copy.deepcopy(fine_stock_data[(fine_stock_data['date'] >= config.train_date_start) & (fine_stock_data['date'] <= config.train_date_end)])
    fsd_train.sort_values(['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
    extra_train_data['fine_stock'] = fsd_train
    fsd_date_lst = fsd_train['date'].unique()
    
    # Validate train fine stock dates match daily dates
    if (len(set(fsd_date_lst) - set(daily_date_lst)) != 0) or (len(set(daily_date_lst) - set(fsd_date_lst)) != 0):
        raise ValueError("[Train, fine stock] | ref date number: {}, extract date number: {} \nmissing dates in ref: {}\nmissing dates in extraction: {}".format(
            len(fsd_date_lst), len(daily_date_lst), set(fsd_date_lst) - set(daily_date_lst), set(daily_date_lst) - set(fsd_date_lst)))
            
    data['extra_train'] = extra_train_data

    # Add validation data if configured
    if (config.valid_date_start is not None) and (config.valid_date_end is not None):
        data = add_validation_fine_data(data, fine_mkt_data, fine_stock_data, config)

    # Add test data if configured
    if (config.test_date_start is not None) and (config.test_date_end is not None):
        data = add_test_fine_data(data, fine_mkt_data, fine_stock_data, config)
        
    return data

def add_validation_fine_data(data, fine_mkt_data, fine_stock_data, config):
    """Add fine data to validation set"""
    daily_date_lst = data['valid']['date'].unique()
    extra_valid_data = {}

    fmd_valid = copy.deepcopy(fine_mkt_data[(fine_mkt_data['date'] >= config.valid_date_start) & (fine_mkt_data['date'] <= config.valid_date_end)])
    fmd_valid.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)
    extra_valid_data['fine_market'] = fmd_valid
    fmd_date_lst = fmd_valid['date'].unique()
    
    # Validate validation fine market dates match daily dates
    if (len(set(fmd_date_lst) - set(daily_date_lst)) != 0) or (len(set(daily_date_lst) - set(fmd_date_lst)) != 0):
        raise ValueError("[Valid, fine market] | ref date number: {}, extract date number: {} \nmissing dates in ref: {}\nmissing dates in extraction: {}".format(
            len(fmd_date_lst), len(daily_date_lst), set(fmd_date_lst) - set(daily_date_lst), set(daily_date_lst) - set(fmd_date_lst)))

    fsd_valid = copy.deepcopy(fine_stock_data[(fine_stock_data['date'] >= config.valid_date_start) & (fine_stock_data['date'] <= config.valid_date_end)])
    fsd_valid.sort_values(['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
    extra_valid_data['fine_stock'] = fsd_valid
    fsd_date_lst = fsd_valid['date'].unique()
    
    # Validate validation fine stock dates match daily dates
    if (len(set(fsd_date_lst) - set(daily_date_lst)) != 0) or (len(set(daily_date_lst) - set(fsd_date_lst)) != 0):
        raise ValueError("[Valid, fine stock] | ref date number: {}, extract date number: {} \nmissing dates in ref: {}\nmissing dates in extraction: {}".format(
            len(fsd_date_lst), len(daily_date_lst), set(fsd_date_lst) - set(daily_date_lst), set(daily_date_lst) - set(fsd_date_lst)))
            
    data['extra_valid'] = extra_valid_data
    return data

def add_test_fine_data(data, fine_mkt_data, fine_stock_data, config):
    """Add fine data to test set"""
    daily_date_lst = data['test']['date'].unique()
    extra_test_data = {}
    
    fmd_test = copy.deepcopy(fine_mkt_data[(fine_mkt_data['date'] >= config.test_date_start) & (fine_mkt_data['date'] <= config.test_date_end)])
    fmd_test.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)
    extra_test_data['fine_market'] = fmd_test
    fmd_date_lst = fmd_test['date'].unique()
    
    # Validate test fine market dates match daily dates
    if (len(set(fmd_date_lst) - set(daily_date_lst)) != 0) or (len(set(daily_date_lst) - set(fmd_date_lst)) != 0):
        raise ValueError("[Test, fine market] | ref date number: {}, extract date number: {} \nmissing dates in ref: {}\nmissing dates in extraction: {}".format(
            len(fmd_date_lst), len(daily_date_lst), set(fmd_date_lst) - set(daily_date_lst), set(daily_date_lst) - set(fmd_date_lst)))

    fsd_test = copy.deepcopy(fine_stock_data[(fine_stock_data['date'] >= config.test_date_start) & (fine_stock_data['date'] <= config.test_date_end)])
    fsd_test.sort_values(['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
    extra_test_data['fine_stock'] = fsd_test
    fsd_date_lst = fsd_test['date'].unique()
    
    # Validate test fine stock dates match daily dates
    if (len(set(fsd_date_lst) - set(daily_date_lst)) != 0) or (len(set(daily_date_lst) - set(fsd_date_lst)) != 0):
        raise ValueError("[Test, fine stock] | ref date number: {}, extract date number: {} \nmissing dates in ref: {}\nmissing dates in extraction: {}".format(
            len(fsd_date_lst), len(daily_date_lst), set(fsd_date_lst) - set(daily_date_lst), set(daily_date_lst) - set(fsd_date_lst)))
            
    data['extra_test'] = extra_test_data
    return data

def generate_market_features(config, freq, daily_date_lst=None):
    """
    Generate market features.
    
    Args:
        config: Configuration object
        freq: Frequency for market data (e.g., '1d', '60m')
        daily_date_lst: Optional list of daily dates
        
    Returns:
        DataFrame with market features
    """
    fpath = os.path.join(config.dataDir, '{}_{}_index.csv'.format(config.market_name, freq))
    isHasFineData = True
    if not os.path.exists(fpath):
        fpath = os.path.join(config.dataDir, '{}_{}_index.csv'.format(config.market_name, '1d'))
        isHasFineData = False
        print("Cannot find the {}-freq market data, will use 1d data instead.".format(freq))
    raw_data = pd.DataFrame(pd.read_csv(fpath, header=0, usecols=['date']+list(config.use_features)))
    raw_data['date'] = pd.to_datetime(raw_data['date'])
    raw_data = raw_data.groupby(['date']).mean().reset_index(drop=False, inplace=False)
    raw_data.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)

    if freq == '1d':
        cur_winsize = config.window_size
    elif freq == '60m':
        cur_winsize = config.fine_window_size
    else:
        raise ValueError("Invalid freq[p1]: {}".format(freq))
    
    close_last_ay = np.array(raw_data['close'])[:-1]
    ma_func = abstract.Function('ma')
    ma_ay = ma_func(np.array(raw_data['close']), timeperiod=cur_winsize+1)
    temp = {'date': np.array(raw_data['date']), 'mkt_{}_close'.format(freq): np.array(raw_data['close']), 'mkt_{}_ma'.format(freq): ma_ay}
    for change_feat in config.use_features:
        cg_ay = np.array(raw_data[change_feat])[1:]
        cg_ay = np.divide(cg_ay, close_last_ay, out=np.ones_like(cg_ay), where=close_last_ay!=0)
        cg_ay[cg_ay==0] = 1
        cg_ay = cg_ay - 1
        cg_ay = np.append([0], cg_ay, axis=0)
        cg_ay = cg_ay * config.feat_scaler
        temp['mkt_{}_{}_w{}'.format(freq, change_feat, 1)] = cg_ay
        for widx in range(2, cur_winsize+1):
            temp['mkt_{}_{}_w{}'.format(freq, change_feat, widx)] = np.append(np.zeros(widx-1), cg_ay[:-(widx-1)], axis=0)
    mkt_pd = pd.DataFrame(temp)

    # Process based on frequency
    if freq == '60m':
        if isHasFineData:
            mkt_pd['time'] = mkt_pd['date'].apply(lambda x: x.strftime('%H:%M:%S')) # Get hour-min-sec
            mkt_pd = mkt_pd[mkt_pd['time']==config.market_close_time[config.market_name]][['date'] + config.finemkt_feat_cols_lst + ['mkt_{}_ma'.format(freq), 'mkt_{}_close'.format(freq)]] # Extract the datapoint of the market close time.
            mkt_pd['date'] = mkt_pd['date'].apply(lambda x: x.strftime('%Y-%m-%d')) # Convert Year-Month-Day hh:mm:ss to Year-Month-Day
            mkt_pd['date'] = pd.to_datetime(mkt_pd['date'])
        else:
            mkt_pd['date'] = pd.to_datetime(mkt_pd['date'])
            mkt_pd = mkt_pd[['date'] + config.finemkt_feat_cols_lst + ['mkt_{}_ma'.format(freq), 'mkt_{}_close'.format(freq)]]
        mkt_pd.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)
    elif freq == '1d':
        pass
    else:
        raise ValueError("Invalid freq[p2]: {}".format(freq))
    
    # columns: date, close/open/high/low_w{1-31/4}
    # The 60m fine market date only includes one datapoint per day (the datapoint is at the market close time), The other timepoints within a day are not included. 
    return mkt_pd

def generate_fine_stock_features(config, daily_date_lst=None):
    """
    Generate fine-grained stock features.
    
    Args:
        config: Configuration object
        daily_date_lst: Optional list of daily dates
        
    Returns:
        DataFrame with fine-grained stock features
    """
    fpath = os.path.join(config.dataDir, '{}_{}_{}.csv'.format(config.market_name, config.topK, config.finefreq))
    isHasFineData = True
    if not os.path.exists(fpath):
        fpath = os.path.join(config.dataDir, '{}_{}_{}.csv'.format(config.market_name, config.topK, '1d'))
        isHasFineData = False
        print("Cannot find the {}-freq stock data, will use 1d data instead.".format(config.finefreq))
    raw_data = pd.DataFrame(pd.read_csv(fpath, header=0, usecols=['date', 'stock']+list(config.use_features)))
    raw_data['date'] = pd.to_datetime(raw_data['date'])
    # raw_data.sort_values(['date', 'stock'], ascending=True, inplace=True, ignore_index=True)
    raw_data = raw_data.groupby(['date', 'stock']).mean().reset_index(drop=False, inplace=False)
    stock_lst = raw_data['stock'].unique()
    fine_data = pd.DataFrame()
    ma_func = abstract.Function('ma')
    
    for stock_id in stock_lst:
        dataSig = copy.deepcopy(raw_data[raw_data['stock']==stock_id])
        dataSig.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)

        ma_ay = ma_func(np.array(dataSig['close']), timeperiod=config.fine_window_size+1)

        temp = {'date': np.array(dataSig['date']), 'stock_{}_close'.format(config.finefreq): np.array(dataSig['close']), 'stock_{}_ma'.format(config.finefreq): ma_ay}
        output_cols = ['date'] + config.finestock_feat_cols_lst + ['stock_{}_ma'.format(config.finefreq), 'stock_{}_close'.format(config.finefreq)]
        
        # Add directional change features if configured
        if config.is_gen_dc_feat:
            dc_events = dc_feature_generation(data=np.array(dataSig['close']), dc_threshold=config.dc_threshold[0])
            temp['stock_{}_dc'.format(config.finefreq)] = dc_events 
            output_cols = output_cols + ['stock_{}_dc'.format(config.finefreq)]
            
        close_last_ay = np.array(dataSig['close'])[:-1]
        # date, stock, close_w1, close_w2, .., open_w1, ..., close/open/high/low_w{1-4}
        for change_feat in config.use_features:
            cg_ay = np.array(dataSig[change_feat])[1:]
            cg_ay = np.divide(cg_ay, close_last_ay, out=np.ones_like(cg_ay), where=close_last_ay!=0)
            cg_ay[cg_ay==0] = 1
            cg_ay = cg_ay - 1
            cg_ay = np.append([0], cg_ay, axis=0)
            cg_ay = cg_ay * config.feat_scaler
            temp['stock_{}_{}_w{}'.format(config.finefreq, change_feat, 1)] = cg_ay
            for widx in range(2, config.fine_window_size+1):
                temp['stock_{}_{}_w{}'.format(config.finefreq, change_feat, widx)] = np.append(np.zeros(widx-1), cg_ay[:-(widx-1)], axis=0)

        temp = pd.DataFrame(temp)
        if isHasFineData:
            temp['time'] = temp['date'].apply(lambda x: x.strftime('%H:%M:%S')) # Get hour-min-sec
            temp = temp[temp['time']==config.market_close_time[config.market_name]][output_cols] # Extract the datapoint of the market close time.
            temp['date'] = temp['date'].apply(lambda x: x.strftime('%Y-%m-%d')) # Convert Year-Month-Day hh:mm:ss to Year-Month-Day
        else:
            temp = temp[output_cols]
        temp.reset_index(drop=True, inplace=True)
        temp['stock'] = stock_id
        fine_data = pd.concat([fine_data, temp], axis=0, join='outer')
        
    fine_data['date'] = pd.to_datetime(fine_data['date'])
    fine_data.sort_values(['stock', 'date'], ascending=True, inplace=True, ignore_index=True)

    # columns: stock, date, close/open/high/low_w{1-4}
    # The date only includes one datapoint per day (the datapoint is at the market close time), The other timepoints within a day are not included. 
    return fine_data