import os
import datetime
import pandas as pd
from typing import Optional

def setup_directories(config_instance, current_date: Optional[str] = None) -> None:
    """Setup directories for results"""
    if current_date is None:
        config_instance.cur_datetime = datetime.datetime.now().strftime('%Y-%m-%d')
    else:
        config_instance.cur_datetime = current_date

    config_instance.res_dir = os.path.join('./res', config_instance.mode, config_instance.rl_model_name,
                                   f'{config_instance.market_name}-{config_instance.topK}', config_instance.cur_datetime)
    os.makedirs(config_instance.res_dir, exist_ok=True)

    config_instance.res_model_dir = os.path.join(config_instance.res_dir, 'model')
    os.makedirs(config_instance.res_model_dir, exist_ok=True)

    config_instance.res_img_dir = os.path.join(config_instance.res_dir, 'graph')
    os.makedirs(config_instance.res_img_dir, exist_ok=True)

def setup_dates(config_instance) -> None:
    """Setup training, validation and testing dates"""
    if config_instance.period_mode in config_instance.date_split_dict:
        date_split = config_instance.date_split_dict[config_instance.period_mode]
    else:
        # Default date split as fallback
        date_split = {
            'train_date_start': '2019-01-10',
            'train_date_end': '2023-12-01',
            'valid_date_start': '2023-12-02',
            'valid_date_end': '2024-12-02',
            'test_date_start': '2024-12-03',
            'test_date_end': '2025-05-03'
        }

    config_instance.train_date_start = pd.Timestamp(date_split['train_date_start'])
    config_instance.train_date_end = pd.Timestamp(date_split['train_date_end'])

    if date_split.get('valid_date_start') and date_split.get('valid_date_end'):
        config_instance.valid_date_start = pd.Timestamp(date_split['valid_date_start'])
        config_instance.valid_date_end = pd.Timestamp(date_split['valid_date_end'])
    else:
        config_instance.valid_date_start = None
        config_instance.valid_date_end = None

    if date_split.get('test_date_start') and date_split.get('test_date_end'):
        config_instance.test_date_start = pd.Timestamp(date_split['test_date_start'])
        config_instance.test_date_end = pd.Timestamp(date_split['test_date_end'])
    else:
        config_instance.test_date_start = None
        config_instance.test_date_end = None

def setup_indicators(config_instance) -> None:
    """Setup technical indicators"""
    # Extract from YAML if available, otherwise use defaults
    config_instance.tech_indicator_talib_lst = []
    config_instance.tech_indicator_extra_lst = ['CHANGE']
    config_instance.tech_indicator_input_lst = config_instance.tech_indicator_talib_lst + config_instance.tech_indicator_extra_lst

    config_instance.dailyRetun_lookback = config_instance.cov_lookback
    config_instance.otherRef_indicator_ma_window = 5
    config_instance.enable_cov_features = False

    config_instance.otherRef_indicator_lst = [
        f'MA-{config_instance.otherRef_indicator_ma_window}',
        f'DAILYRETURNS-{config_instance.dailyRetun_lookback}'
    ] 