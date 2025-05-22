import yaml
from typing import Optional

def generate_config_summary_string(config_instance) -> str:
    """Generate a string summary of the configuration for printing."""
    log_str = '=' * 30 + '\n'
    para_str = f'{config_instance.notes} \n'
    log_str = log_str + para_str
    para_str = f'mode: {config_instance.mode}, rl_model_name: {config_instance.rl_model_name}, market_name: {config_instance.market_name}, topK: {config_instance.topK}, dataDir: {config_instance.dataDir}, enable_controller: {config_instance.enable_controller}, \n'
    log_str = log_str + para_str
    para_str = f'trade_pattern: {config_instance.trade_pattern} \n'
    log_str = log_str + para_str
    para_str = f'period_mode: {config_instance.period_mode}, num_epochs: {config_instance.num_epochs}, cov_lookback: {config_instance.cov_lookback}, norm_method: {config_instance.norm_method}, benchmark_algo: {config_instance.benchmark_algo}, trained_best_model_type: {config_instance.trained_best_model_type}, pricePredModel: {config_instance.pricePredModel}, \n'
    log_str = log_str + para_str
    para_str = f'is_enable_dynamic_risk_bound: {config_instance.is_enable_dynamic_risk_bound}, risk_market: {config_instance.risk_market}, risk_default: {config_instance.risk_default}, cbf_gamma: {config_instance.cbf_gamma}, ars_trial: {config_instance.ars_trial} \n'
    log_str = log_str + para_str
    para_str = f'cur_datetime: {config_instance.cur_datetime}, res_dir: {config_instance.res_dir}, tradeDays_per_year: {config_instance.tradeDays_per_year}, tradeDays_per_month: {config_instance.tradeDays_per_month}, seed_num: {config_instance.seed_num}, \n'
    log_str = log_str + para_str
    para_str = f'train_date_start: {config_instance.train_date_start}, train_date_end: {config_instance.train_date_end}, valid_date_start: {config_instance.valid_date_start}, valid_date_end: {config_instance.valid_date_end}, test_date_start: {config_instance.test_date_start}, test_date_end: {config_instance.test_date_end}, \n'
    log_str = log_str + para_str
    para_str = f'tech_indicator_input_lst: {config_instance.tech_indicator_input_lst}, \n'
    log_str = log_str + para_str
    para_str = f'otherRef_indicator_lst: {config_instance.otherRef_indicator_lst}, enable_cov_features: {config_instance.enable_cov_features} \n'
    log_str = log_str + para_str
    para_str = f'tmp_name: {config_instance.tmp_name}, mkt_rf: {config_instance.mkt_rf} \n'
    log_str = log_str + para_str
    para_str = f'invest_env_para: {config_instance.invest_env_para}, \n'
    log_str = log_str + para_str
    para_str = f'model_para: {config_instance.model_para}, \n'
    log_str = log_str + para_str
    para_str = f'only_long_algo_lst: {config_instance.only_long_algo_lst}, \n'
    log_str = log_str + para_str
    para_str = f'lstr para: use_features: {config_instance.use_features}, window_size: {config_instance.window_size}, freq: {config_instance.freq}, finefreq: {config_instance.finefreq}, fine_window_size: {config_instance.fine_window_size}, \n'
    log_str = log_str + para_str
    para_str = f'enable_market_observer: {config_instance.enable_market_observer}, mktobs_algo: {config_instance.mktobs_algo}, feat_scaler: {config_instance.feat_scaler} \n'
    log_str = log_str + para_str
    log_str = log_str + '=' * 30 + '\n'
    return log_str

def get_config_as_dict(config_instance) -> dict:
    """Convert current configuration to a dictionary for saving."""
    config_dict = {
        'notes': config_instance.notes,
        'benchmark_algo': config_instance.benchmark_algo,
        'market_name': config_instance.market_name,
        'topK': config_instance.topK,
        'num_epochs': config_instance.num_epochs,
        'dataDir': config_instance.dataDir,
        'seed_num': config_instance.seed_num,
        'trade_pattern': config_instance.trade_pattern,
        'lambda_1': config_instance.lambda_1,
        'lambda_2': config_instance.lambda_2,
        'risk_default': config_instance.risk_default,
        'risk_up_bound': config_instance.risk_up_bound,
        'risk_hold_bound': config_instance.risk_hold_bound,
        'risk_down_bound': config_instance.risk_down_bound,
        'risk_market': config_instance.risk_market,
        'cbf_gamma': config_instance.cbf_gamma,
        'period_mode': config_instance.period_mode,
        'date_split': config_instance.date_split_dict,
        'pricePredModel': config_instance.pricePredModel,
        'cov_lookback': config_instance.cov_lookback,
        'norm_method': config_instance.norm_method,
        'tradeDays_per_year': config_instance.tradeDays_per_year,
        'tradeDays_per_month': config_instance.tradeDays_per_month,
        'mkt_rf': config_instance.mkt_rf,
        'market_close_time': config_instance.market_close_time,
        'invest_env': {
            'max_shares': config_instance.invest_env_para['max_shares'],
            'initial_asset': config_instance.invest_env_para['initial_asset'],
            'transaction_cost': config_instance.invest_env_para['transaction_cost'],
            'slippage': config_instance.invest_env_para['slippage']
        },
        'rl': {
            'reward_scaling': config_instance.reward_scaling,
            'learning_rate': config_instance.learning_rate,
            'batch_size': config_instance.batch_size,
            'gradient_steps': config_instance.gradient_steps,
            'train_freq': config_instance.train_freq,
            'ars_trial': config_instance.ars_trial
        },
        'model': {
            'window_size': config_instance.window_size,
            'po_lr': config_instance.po_lr,
            'po_weight_decay': config_instance.po_weight_decay,
            'use_features': config_instance.use_features
        },
        'market_observer': {
            'freq': config_instance.freq,
            'finefreq': config_instance.finefreq,
            'fine_window_size': config_instance.fine_window_size,
            'feat_scaler': config_instance.feat_scaler,
            'hidden_vec_loss_weight': config_instance.hidden_vec_loss_weight,
            'sigma_loss_weight': config_instance.sigma_loss_weight,
            'lambda_min': config_instance.lambda_min,
            'lambda_max': config_instance.lambda_max,
            'sigma_min': config_instance.sigma_min,
            'sigma_max': config_instance.sigma_max,
            'dc': {
                'threshold': config_instance.dc_threshold
            }
        },
        'only_long_algo_lst': config_instance.only_long_algo_lst,
        'use_cash_algo_lst': config_instance.use_cash_algo_lst
    }
    return config_dict

def save_config_to_yaml(config_instance, output_path: Optional[str] = None) -> None:
    """Save current configuration to a YAML file."""
    if output_path is None:
        output_path = config_instance.config_path
    
    config_dict = get_config_as_dict(config_instance)
    
    with open(output_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False) 