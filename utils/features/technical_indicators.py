#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         technical_indicators.py
 Description:  Technical feature indicator generation functions
 Author:       MASA
---------------------------------
'''
import numpy as np
import pandas as pd
import copy
from talib import abstract
# Update this import
from utils.features.directional_change import dc_feature_generation

def generate_technical_features(data, config):
    """
    Generate technical indicators for each stock.
    
    Args:
        data: DataFrame with stock data
        config: Configuration object with technical indicator parameters
    
    Returns:
        DataFrame with generated technical features
    """
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
    # ['date', 'stock', 'open', 'high', 'low', 'close', 'volume']
    rawColLst = list(data.columns)
    datax = copy.deepcopy(data)
    stock_lst = datax['stock'].unique()
    for indidx, sigIndicatorName in enumerate(list(config.tech_indicator_input_lst) + list(config.otherRef_indicator_lst)):
        if sigIndicatorName.split('-')[0] in ['DAILYRETURNS']: 
            continue
        ind_df = pd.DataFrame()
        for sigStockName in stock_lst:
            dataSig = copy.deepcopy(data[data['stock']==sigStockName])
            dataSig.sort_values(['date'], ascending=True, inplace=True, ignore_index=True)
            if sigIndicatorName == 'CHANGE':
                temp = generate_change_indicator(dataSig, config)
            # {indicator name}-{window}-{output field}-{input field}
            elif (sigIndicatorName in config.tech_indicator_talib_lst) or (sigIndicatorName in config.otherRef_indicator_lst):
                temp = generate_talib_indicator(dataSig, sigIndicatorName, config)
            else:
                raise ValueError("Please specify the category of the indicator: {}".format(sigIndicatorName))               
            
            temp = pd.DataFrame(temp)
            temp['stock'] = sigStockName
            temp['date'] = np.array(dataSig['date'])
            ind_df = pd.concat([ind_df, temp], axis=0, join='outer')
        datax = pd.merge(datax, ind_df, how='outer', on=['stock', 'date'])
    
    datax.sort_values(['stock', 'date'], ascending=True, inplace=True, ignore_index=True)
    techIndicatorLst = sorted(list(set(datax.columns) - set(rawColLst) - set(config.otherRef_indicator_lst)))
    
    return datax, rawColLst, techIndicatorLst

def generate_change_indicator(dataSig, config):
    """Generate CHANGE indicator features"""
    temp = {}
    # Generate the training features
    cg_close_ay_last = np.array(dataSig['close'])[:-1]
    for change_feat in config.use_features:
        cg_ay = np.array(dataSig[change_feat])[1:]
        cg_ay = np.divide(cg_ay, cg_close_ay_last, out=np.ones_like(cg_ay), where=cg_close_ay_last!=0)
        cg_ay[cg_ay==0] = 1
        cg_ay = cg_ay - 1 # -> mean=0
        # fill the first day data
        cg_ay = np.append([0], cg_ay, axis=0)
        temp['{}_w{}'.format(change_feat, 1)] = cg_ay
        for widx in range(2, config.window_size+1):
            temp['{}_w{}'.format(change_feat, widx)] = np.append(np.zeros(widx-1), cg_ay[:-(widx-1)], axis=0)
    return temp

def generate_talib_indicator(dataSig, sigIndicatorName, config):
    """Generate technical indicators using TA-Lib"""
    indNameLst = sigIndicatorName.split('-')
    indFunc = abstract.Function(indNameLst[0])
    output_fields = indFunc.output_names
    if 'price' in indFunc.input_names.keys():
        ori_ifield = indFunc.input_names['price']
    if 'prices' in indFunc.input_names.keys():
        ori_ifield = indFunc.input_names['prices']
    
    if len(indNameLst) == 1:
        iname = sigIndicatorName
        window_size = None
        ifield = ori_ifield
        ofield = None
    elif len(indNameLst) == 2:
        iname = indNameLst[0]
        if indNameLst[1] == 'None':
            window_size = None
        else:
            window_size = int(indNameLst[1])
        ofield = None
        ifield = ori_ifield
    elif len(indNameLst) == 3:
        iname = indNameLst[0]
        if indNameLst[1] == 'None':
            window_size = None
        else:
            window_size = int(indNameLst[1])
        if indNameLst[2] == 'None':
            ofield = None
        else:
            ofield = indNameLst[2]
        ifield = ori_ifield
    elif len(indNameLst) == 4:
        iname = indNameLst[0]
        if indNameLst[1] == 'None':
            window_size = None
        else:
            window_size = int(indNameLst[1])
        if indNameLst[2] == 'None':
            ofield = None
        else:
            ofield = indNameLst[2]
        if indNameLst[3] == 'None':
            ifield = ori_ifield
        else:
            ifield = indNameLst[3]   
    else:
        raise ValueError("Unexpect indicator {}".format(sigIndicatorName))
    
    if iname in ['OBV']:
        ind_val = indFunc(dataSig[['open', 'high', 'low', 'close', 'volume']])
    elif 'price' in indFunc.input_names.keys():
        ind_val = indFunc(dataSig[['open', 'high', 'low', 'close', 'volume']], timeperiod=window_size, price=ifield)
    elif 'prices' in indFunc.input_names.keys():
        ind_val = indFunc(dataSig[['open', 'high', 'low', 'close', 'volume']], timeperiod=window_size, prices=ifield)
    else:
        raise ValueError("Invalid input fields: {}".format(indFunc.input_names))

    if len(output_fields) == 1:
        temp = {sigIndicatorName: np.array(ind_val.values)}
    else:
        if ofield is None:
            if sigIndicatorName == 'MACD':
                temp = {sigIndicatorName: np.array(ind_val['macd'])}
            elif sigIndicatorName == 'AROON':
                temp = {'AROONDOWN': np.array(ind_val['aroondown']), 'AROONUP': np.array(ind_val['aroonup'])}
            elif sigIndicatorName == 'BBANDS':
                temp = {'BOLLUP': np.array(ind_val['upperband']), 'BOLLMID': np.array(ind_val['middleband']), 'BOLLLOW': np.array(ind_val['lowerband'])}
            else:
                temp = {sigIndicatorName: np.array(ind_val[sorted(list(ind_val.keys()))[0]])}
        else:
            temp = {sigIndicatorName: np.array(ind_val[ofield])}
            
    return temp