import yaml

def load_yaml_config(config_instance, config_path: str, seed_num: int):
    """Load configuration from YAML file and set initial attributes."""
    config_instance.config_path = config_path
    try:
        with open(config_instance.config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Load general settings
        config_instance.notes = config.get('notes', 'AAMAS MASA Implementation')
        config_instance.benchmark_algo = config.get('benchmark_algo', 'MASA-dc')
        config_instance.market_name = config.get('market_name', 'DJIA')
        config_instance.topK = config.get('topK', 5)
        config_instance.num_epochs = config.get('num_epochs', 1)

        # Load directory settings
        config_instance.dataDir = config.get('dataDir', './data')
        config_instance.seed_num = config.get('seed_num', 2022)

        # Override seed if provided
        if seed_num is not None:
            config_instance.seed_num = seed_num

        # Load trading parameters
        config_instance.trade_pattern = config.get('trade_pattern', 1)
        config_instance.lambda_1 = config.get('lambda_1', 1000.0)
        config_instance.lambda_2 = config.get('lambda_2', 10.0)
        config_instance.risk_default = config.get('risk_default', 0.017)
        config_instance.risk_up_bound = config.get('risk_up_bound', 0.012)
        config_instance.risk_hold_bound = config.get('risk_hold_bound', 0.014)
        config_instance.risk_down_bound = config.get('risk_down_bound', 0.017)
        config_instance.risk_market = config.get('risk_market', 0.001)
        config_instance.cbf_gamma = config.get('cbf_gamma', 0.7)

        # Load period mode
        config_instance.period_mode = config.get('period_mode', 1)
        config_instance.date_split_dict = config.get('date_split', {})

        # Load financial parameters
        config_instance.tradeDays_per_year = config.get('tradeDays_per_year', 252)
        config_instance.tradeDays_per_month = config.get('tradeDays_per_month', 21)
        config_instance.mkt_rf = config.get('mkt_rf', {'SP500': 1.6575, 'CSI300': 3.037, 'DJIA': 1.6575})
        config_instance.market_close_time = config.get('market_close_time', {'CSI300': '15:00:00'})

        # Load investment environment parameters
        invest_env = config.get('invest_env', {})
        config_instance.invest_env_para = {
            'max_shares': invest_env.get('max_shares', 100),
            'initial_asset': invest_env.get('initial_asset', 1000000),
            'transaction_cost': invest_env.get('transaction_cost', 0.0003),
            'slippage': invest_env.get('slippage', 0.001),
            'seed_num': config_instance.seed_num,
            'reward_scaling': config.get('rl', {}).get('reward_scaling', 1),
            'norm_method': config.get('norm_method', 'sum')
        }

        # Load RL parameters
        rl_config = config.get('rl', {})
        config_instance.reward_scaling = rl_config.get('reward_scaling', 1)
        config_instance.learning_rate = rl_config.get('learning_rate', 0.0001)
        config_instance.batch_size = rl_config.get('batch_size', 50)
        config_instance.gradient_steps = rl_config.get('gradient_steps', 1)
        config_instance.train_freq = rl_config.get('train_freq', [1, 'episode'])
        config_instance.ars_trial = rl_config.get('ars_trial', 10)

        # Load algorithm lists
        config_instance.only_long_algo_lst = config.get('only_long_algo_lst', ['CRP', 'EG', 'OLMAR', 'PAMR', 'RMR'])
        config_instance.use_cash_algo_lst = config.get('use_cash_algo_lst', ['RAT', 'EIIE', 'PPN'])

        # Load price prediction model
        config_instance.pricePredModel = config.get('pricePredModel', 'MA')
        config_instance.cov_lookback = config.get('tech_indicator', {}).get('dailyReturn_lookback', 5)
        config_instance.norm_method = config.get('norm_method', 'sum')

        # Market observer config
        market_observer = config.get('market_observer', {})
        config_instance.dc_threshold = market_observer.get('dc', {}).get('threshold', [0.01])

    except FileNotFoundError:
        print(f"Configuration file not found: {config_instance.config_path}")
        print("Using default configuration values.")
        # Set defaults for critical values
        config_instance.notes = 'AAMAS MASA Implementation'
        config_instance.benchmark_algo = 'MASA-dc'
        config_instance.market_name = 'DJIA'
        config_instance.topK = 5
        config_instance.num_epochs = 1
        config_instance.dataDir = './data'
        config_instance.seed_num = 2022 if seed_num is None else seed_num
        config_instance.period_mode = 1
        config_instance.pricePredModel = 'MA'
        config_instance.cov_lookback = 5
        config_instance.norm_method = 'sum'
        config_instance.risk_default = 0.017
        config_instance.risk_up_bound = 0.012
        config_instance.risk_hold_bound = 0.014
        config_instance.risk_down_bound = 0.017
        config_instance.risk_market = 0.001
        config_instance.cbf_gamma = 0.7
        config_instance.reward_scaling = 1
        config_instance.learning_rate = 0.0001
        config_instance.batch_size = 50
        config_instance.gradient_steps = 1
        config_instance.train_freq = [1, 'episode']
        config_instance.ars_trial = 10
        config_instance.trade_pattern = 1
        config_instance.lambda_1 = 1000.0
        config_instance.lambda_2 = 10.0
        config_instance.tradeDays_per_year = 252
        config_instance.tradeDays_per_month = 21
        config_instance.mkt_rf = {'SP500': 1.6575, 'CSI300': 3.037, 'DJIA': 1.6575}
        config_instance.market_close_time = {'CSI300': '15:00:00'}
        config_instance.invest_env_para = {
            'max_shares': 100,
            'initial_asset': 1000000,
            'reward_scaling': 1,
            'norm_method': 'sum',
            'transaction_cost': 0.0003,
            'slippage': 0.001,
            'seed_num': config_instance.seed_num
        }
        config_instance.only_long_algo_lst = ['CRP', 'EG', 'OLMAR', 'PAMR', 'RMR']
        config_instance.use_cash_algo_lst = ['RAT', 'EIIE', 'PPN']
        config_instance.dc_threshold = [0.01]

        # Default date split dictionary
        config_instance.date_split_dict = {
            1: {
                'train_date_start': '2019-01-10',
                'train_date_end': '2023-12-01',
                'valid_date_start': '2023-12-02',
                'valid_date_end': '2024-12-02',
                'test_date_start': '2024-12-03',
                'test_date_end': '2025-05-03'
            }
        } 