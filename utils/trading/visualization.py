#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         visualization.py
 Description:  Saving and visualization functions for trading environment.
 Author:       MASA
---------------------------------
'''
import os
import pandas as pd
import numpy as np

def save_action_memory(env):
    """
    Save action memory to dataframe.
    
    Args:
        env: Trading environment instance
        
    Returns:
        DataFrame of action memory
    """
    action_pd = pd.DataFrame(np.array(env.actions_memory), columns=env.stock_lst)
    action_pd['date'] = env.date_memory
    return action_pd

def save_profile(env, invest_profile):
    """
    Save investment profile to file.
    
    Args:
        env: Trading environment instance
        invest_profile: Dictionary with investment results
    """
    # Save basic data to profile history
    for fname in env.profile_hist_field_lst:
        if fname in list(invest_profile.keys()):
            env.profile_hist_ep[fname].append(invest_profile[fname])
        else:
            raise ValueError('Cannot find the field [{}] in invest profile..'.format(fname))
    
    # Save profile history to CSV
    phist_df = pd.DataFrame(env.profile_hist_ep, columns=env.profile_hist_field_lst)
    phist_df.to_csv(os.path.join(env.config.res_dir, '{}_profile.csv'.format(env.mode)), index=False)

    # Calculate average times
    cputime_avg = np.mean(phist_df['cputime'])
    systime_avg = np.mean(phist_df['systime'])

    # Find best model by selection criteria
    bestmodel_dict = find_best_model(env, phist_df)
    
    # Log performance
    if True:
        print("-"*30)
        log_str = "Mode: {}, Ep: {}, Current epoch capital: {}, historical best captial ({} ep): {} | solvable: {}, insolvable: {} | step count: {} | cputime cur: {} s, avg: {} s, system time cur: {} s/ep, avg: {} s/ep..".format(
            env.mode, env.epoch, env.cur_capital, 
            bestmodel_dict['{}_ep'.format(env.config.trained_best_model_type)], 
            bestmodel_dict[env.config.trained_best_model_type], 
            np.array(phist_df['solver_solvable'])[-1], 
            np.array(phist_df['solver_insolvable'])[-1], 
            env.stepcount, 
            np.round(np.array(phist_df['cputime'])[-1], 2), 
            np.round(cputime_avg, 2), 
            np.round(np.array(phist_df['systime'])[-1], 2), 
            np.round(systime_avg, 2)
        )
        print(log_str)
    
    # Save best model info
    bestmodel_df = pd.DataFrame([bestmodel_dict])
    bestmodel_df.to_csv(os.path.join(env.config.res_dir, '{}_bestmodel.csv'.format(env.mode)), index=False)

    # Save step data for different policy versions
    save_step_data(env, invest_profile, bestmodel_dict)

def find_best_model(env, phist_df):
    """
    Find the best model based on the selected criteria.
    
    Args:
        env: Trading environment instance
        phist_df: DataFrame with performance history
        
    Returns:
        Dictionary with best model info
    """
    bestmodel_dict = {}
    
    if env.config.trained_best_model_type == 'max_capital':
        field_name = 'final_capital'
        v = np.max(phist_df[field_name])  # Maximum value will be recorded
    elif 'loss' in env.config.trained_best_model_type:
        field_name = 'reward_sum'
        v = np.max(phist_df[field_name])  # Maximum value will be recorded
    elif env.config.trained_best_model_type == 'sharpeRatio':
        field_name = 'sharpeRatio'
        v = np.max(phist_df[field_name])  # Maximum value will be recorded
    elif env.config.trained_best_model_type == 'volatility':
        field_name = 'volatility'
        v = np.min(phist_df[field_name])  # Minimum value will be recorded
    elif env.config.trained_best_model_type == 'mdd':
        field_name = 'mdd'
        v = np.min(phist_df[field_name])  # Minimum value will be recorded
    else:
        raise ValueError('Unknown implementation with the best model type [{}]..'.format(env.config.trained_best_model_type))
    
    v_ep = list(phist_df[phist_df[field_name]==v]['ep'])[0]
    bestmodel_dict['{}_ep'.format(env.config.trained_best_model_type)] = v_ep
    bestmodel_dict[env.config.trained_best_model_type] = v
    
    return bestmodel_dict

def save_step_data(env, invest_profile, bestmodel_dict):
    """
    Save step-by-step data for different policy versions.
    
    Args:
        env: Trading environment instance
        invest_profile: Dictionary with investment results
        bestmodel_dict: Dictionary with best model info
    """
    fpath = os.path.join(env.config.res_dir, '{}_stepdata.csv'.format(env.mode))
    
    # First policy or load existing
    if not os.path.exists(fpath):
        step_data = create_first_policy_data(invest_profile)
        step_data = pd.DataFrame(step_data)
    else:
        step_data = pd.DataFrame(pd.read_csv(fpath, header=0))
    
    # Best policy
    if bestmodel_dict['{}_ep'.format(env.config.trained_best_model_type)] == invest_profile['ep']:
        step_data = update_step_data(step_data, invest_profile, 'best')
    
    # Valid best policy (for test mode)
    if env.mode == 'test':
        valid_fpath = os.path.join(env.config.res_dir, 'valid_bestmodel.csv')
        if os.path.exists(valid_fpath):
            valid_records = pd.DataFrame(pd.read_csv(valid_fpath, header=0))
            if int(valid_records['{}_ep'.format(env.config.trained_best_model_type)][0]) == invest_profile['ep']:
                step_data = update_step_data(step_data, invest_profile, 'validbest')
                
                # Log performance for validation best model
                print("-"*30)
                log_str = "Mode: Best-{}, Ep: {}, Capital (test set, by using the best validation model, {} ep): {} ".format(
                    env.mode, env.epoch, 
                    int(valid_records['{}_ep'.format(env.config.trained_best_model_type)][0]), 
                    np.array(step_data['capital_policy_validbest'])[-1]
                )
                print(log_str)

    # Last policy (at end of training)
    if invest_profile['ep'] == env.config.num_epochs:
        step_data = update_step_data(step_data, invest_profile, 'last')
    
    # Save step data
    step_data.to_csv(fpath, index=False)

def create_first_policy_data(invest_profile):
    """Create data dictionary for first policy"""
    return {
        'capital_policy_1': invest_profile['asset_lst'], 
        'dailyReturn_policy_1': invest_profile['daily_return_lst'],
        'reward_policy_1': invest_profile['reward_lst'], 
        'strategyVolatility_policy_1': invest_profile['stg_vol_lst'],
        'risk_policy_1': invest_profile['risk_lst'], 
        'risk_wocbf_policy_1': invest_profile['risk_wocbf_lst'], 
        'capital_wocbf_policy_1': invest_profile['capital_wocbf_lst'],
        'dailySR_policy_1': invest_profile['daily_sr_lst'], 
        'dailySR_wocbf_policy_1': invest_profile['daily_sr_wocbf_lst'], 
        'riskAccepted_policy_1': invest_profile['risk_adj_lst'],
        'ctrlWeight_policy_1': invest_profile['ctrl_weight_lst'],
        'solvable_flag_policy_1': invest_profile['solvable_flag'],
        'risk_pred_policy_1': invest_profile['risk_pred_lst'],
        'final_action_abssum_policy_1': invest_profile['final_action_abssum_lst'],
        'rl_action_abssum_policy_1': invest_profile['rl_action_abssum_lst'],
        'cbf_action_abssum_policy_1': invest_profile['cbf_action_abssum_lst'],
        'downsideAtVol_risk_policy_1': invest_profile['daily_downsideAtVol_risk_lst'],
        'downsideAtValue_risk_policy_1': invest_profile['daily_downsideAtValue_risk_lst'],
        'cvar_policy_1': invest_profile['cvar_lst'], 
        'cvar_raw_policy_1': invest_profile['cvar_raw_lst'],
    }

def update_step_data(step_data, invest_profile, policy_suffix):
    """Update step data for a specific policy type"""
    fields = [
        'capital', 'dailyReturn', 'reward', 'strategyVolatility', 'risk', 
        'risk_wocbf', 'capital_wocbf', 'dailySR', 'dailySR_wocbf', 
        'riskAccepted', 'ctrlWeight', 'solvable_flag', 'risk_pred', 
        'final_action_abssum', 'rl_action_abssum', 'cbf_action_abssum', 
        'downsideAtVol_risk', 'downsideAtValue_risk', 'cvar', 'cvar_raw'
    ]
    
    source_fields = [
        'asset_lst', 'daily_return_lst', 'reward_lst', 'stg_vol_lst', 'risk_lst', 
        'risk_wocbf_lst', 'capital_wocbf_lst', 'daily_sr_lst', 'daily_sr_wocbf_lst', 
        'risk_adj_lst', 'ctrl_weight_lst', 'solvable_flag', 'risk_pred_lst', 
        'final_action_abssum_lst', 'rl_action_abssum_lst', 'cbf_action_abssum_lst', 
        'daily_downsideAtVol_risk_lst', 'daily_downsideAtValue_risk_lst', 'cvar_lst', 'cvar_raw_lst'
    ]
    
    for field, source_field in zip(fields, source_fields):
        step_data[f'{field}_policy_{policy_suffix}'] = invest_profile[source_field]
    
    return step_data