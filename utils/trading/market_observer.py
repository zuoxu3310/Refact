#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         market_observer.py
 Description:  Market observer integration for trading environment.
 Author:       MASA
---------------------------------
'''
import numpy as np

def run_market_observer(env, stage=None, rate_of_price_change=None):
    """
    Run market observer to get risk boundary and stock MA prices.
    
    Args:
        env: Trading environment instance
        stage: Stage of execution ('init', 'reset', 'run')
        rate_of_price_change: Array of price change rates
        
    Returns:
        Tuple of current risk boundary and stock MA prices
    """
    cur_date = env.curData['date'].unique()[0]
    
    if env.config.enable_market_observer:
        # Reset market observer if needed
        if stage in ['reset', 'init'] and (env.mode == 'train'):
            env.mkt_observer.reset()

        # Get market features
        finemkt_feat = env.extra_data['fine_market']
        ma_close = finemkt_feat[finemkt_feat['date']==cur_date][['mkt_{}_close'.format(env.config.finefreq), 'mkt_{}_ma'.format(env.config.finefreq)]].values[-1]
        mkt_cur_close_price = ma_close[0]
        mkt_ma_price = ma_close[1]
        finemkt_feat = finemkt_feat[finemkt_feat['date']==cur_date][env.config.finemkt_feat_cols_lst].values
        finemkt_feat = np.reshape(finemkt_feat, (len(env.config.use_features), env.config.fine_window_size)) # -> (features, window_size)
        finemkt_feat = np.expand_dims(finemkt_feat, axis=0) # -> (batch=1, features, window_size)
        
        # Update market observer with rewards if in training mode
        if (rate_of_price_change is not None) and (env.mode == 'train'):
            if mkt_cur_close_price > env.mkt_last_close_price:
                mkt_direction = 0  # Up
            elif mkt_cur_close_price < env.mkt_last_close_price:
                mkt_direction = 2  # Down
            else:
                mkt_direction = 1  # No change
            mkt_direction = np.array([mkt_direction])
            env.mkt_observer.update_hidden_vec_reward(mode=env.mode, rate_of_price_change=rate_of_price_change, mkt_direction=mkt_direction)

        # Get stock features
        finestock_feat = env.extra_data['fine_stock']
        stock_cur_close_price = finestock_feat[finestock_feat['date']==cur_date]['stock_{}_close'.format(env.config.finefreq)].values
        stock_ma_price = finestock_feat[finestock_feat['date']==cur_date]['stock_{}_ma'.format(env.config.finefreq)].values
        
        # Get directional change events if enabled
        if env.config.is_gen_dc_feat:
            dc_events = finestock_feat[finestock_feat['date']==cur_date]['stock_{}_dc'.format(env.config.finefreq)].values
        else:
            dc_events = None
        
        # Process stock features
        finestock_feat = finestock_feat[finestock_feat['date']==cur_date][env.config.finestock_feat_cols_lst].values
        finestock_feat = np.reshape(finestock_feat, (env.config.topK, len(env.config.use_features), env.config.fine_window_size))
        finestock_feat = np.transpose(finestock_feat, (1, 0, 2))  # -> (features, num_of_stocks, window_size)
        finestock_feat = np.expand_dims(finestock_feat, axis=0)  # -> (batch=1, features, num_of_stocks, window_size)
        
        # Prepare input kwargs
        input_kwargs = {
            'mode': env.mode, 
            'stock_ma_price': np.array([stock_ma_price]), 
            'stock_cur_close_price': np.array([stock_cur_close_price]), 
            'dc_events': np.array([dc_events])
        }

        # Get predictions from market observer
        cur_hidden_vector_ay, lambda_val, sigma_val = env.mkt_observer.predict(
            finemkt_feat=finemkt_feat, 
            finestock_feat=finestock_feat, 
            **input_kwargs
        )
        
        # Determine risk boundary based on market direction
        if env.config.is_enable_dynamic_risk_bound:
            if int(sigma_val[-1]) == 0:
                # Up trend
                cur_risk_boundary = env.config.risk_up_bound
            elif int(sigma_val[-1]) == 1:
                # Sideways/Hold
                cur_risk_boundary = env.config.risk_hold_bound
            elif int(sigma_val[-1]) == 2:
                # Down trend
                cur_risk_boundary = env.config.risk_down_bound
            else:
                raise ValueError('Unknown sigma value [{}]..'.format(sigma_val[-1]))
        else:
            cur_risk_boundary = env.config.risk_default
        
        # Update state with hidden vector
        env.state = np.append(env.state, cur_hidden_vector_ay[-1], axis=0)
        env.mkt_last_close_price = mkt_cur_close_price
    else:
        # Market observer disabled - use default risk boundary
        cur_risk_boundary = env.config.risk_default
        
        # Get stock MA prices for controller mode
        if env.config.mode == 'RLcontroller':
            finestock_feat = env.extra_data['fine_stock']
            stock_ma_price = finestock_feat[finestock_feat['date']==cur_date]['stock_{}_ma'.format(env.config.finefreq)].values
        else:
            stock_ma_price = None

    return cur_risk_boundary, stock_ma_price