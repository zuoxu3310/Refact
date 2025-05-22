from RL_controller.TD3_controller import TD3PolicyOriginal

def load_model_config(config_instance) -> None:
    """Load model configuration"""
    # These could come from YAML in a more complete implementation
    config_instance.use_features = ['close', 'open', 'high', 'low']
    config_instance.window_size = 31
    config_instance.po_lr = 0.0001
    config_instance.po_weight_decay = 0.001

def load_market_observer_config(config_instance) -> None:
    """Load market observer configuration"""
    config_instance.freq = '1d'
    config_instance.finefreq = '60m'
    config_instance.fine_window_size = 4
    config_instance.feat_scaler = 10

    config_instance.hidden_vec_loss_weight = 1e4
    config_instance.sigma_loss_weight = 1e5
    config_instance.lambda_min = 0.0
    config_instance.lambda_max = 1.0
    config_instance.sigma_min = 0.0
    config_instance.sigma_max = 1.0

    config_instance.finestock_feat_cols_lst = []
    config_instance.finemkt_feat_cols_lst = []
    for ifeat in config_instance.use_features:
        for iwin in range(1, config_instance.fine_window_size + 1):
            config_instance.finestock_feat_cols_lst.append(f'stock_{config_instance.finefreq}_{ifeat}_w{iwin}')
            config_instance.finemkt_feat_cols_lst.append(f'mkt_{config_instance.finefreq}_{ifeat}_w{iwin}')

def load_parameters(config_instance) -> None:
    """Load model parameters"""
    # Determine policy name based on configuration
    if config_instance.enable_market_observer:
        if config_instance.rl_model_name == 'TD3':
            policy_name = 'TD3PolicyAdj'
        else:
            raise ValueError(f"Cannot specify the {config_instance.rl_model_name} policy name when enabling market observer.")
    else:
        if config_instance.rl_model_name == 'TD3':
            policy_name = TD3PolicyOriginal
        else:
            if config_instance.mode in ['RLonly', 'RLcontroller']:
                raise ValueError(f"Cannot specify the {config_instance.rl_model_name} policy name when using stable-baseline.")
            else:
                policy_name = "MlpPolicy"

    # Base parameters for all models
    base_para = {
        'policy': policy_name,
        'learning_rate': config_instance.learning_rate,
        'buffer_size': 1000000,
        'learning_starts': 100,
        'batch_size': config_instance.batch_size,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': (config_instance.train_freq[0], config_instance.train_freq[1]),
        'gradient_steps': config_instance.gradient_steps,
        'action_noise': None,
        'replay_buffer_class': None,
        'replay_buffer_kwargs': None,
        'optimize_memory_usage': False,
        'tensorboard_log': None,
        'policy_kwargs': None,
        'verbose': 1,
        'seed': config_instance.seed_num,
        'device': 'auto',
        '_init_setup_model': True,
    }

    # Algorithm-specific parameters
    algo_para = {
        'TD3': {
            'policy_delay': 2,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
        },
        'SAC': {
            'ent_coef': 'auto',
            'target_update_interval': 1,
            'target_entropy': 'auto',
            'use_sde': False,
            'sde_sample_freq': -1,
            'use_sde_at_warmup': False,
        },
        'PPO': {
            'n_steps': 100
        },
    }

    # Adjust network architecture for larger portfolios
    if (config_instance.topK == 20) or (config_instance.topK == 30):
        algo_para['TD3']['policy_kwargs'] = {
            'net_arch': [1024, 512, 128],
        }

    # Parameters to remove for specific algorithms
    algo_para_rm_from_base = {
        'PPO': [
            'buffer_size', 'learning_starts', 'tau', 'train_freq',
            'gradient_steps', 'action_noise', 'replay_buffer_class',
            'replay_buffer_kwargs', 'optimize_memory_usage'
        ]
    }

    # Combine parameters
    if config_instance.rl_model_name in algo_para:
        config_instance.model_para = {**base_para, **algo_para[config_instance.rl_model_name]}
    else:
        config_instance.model_para = base_para

    # Remove unnecessary parameters for specific algorithms
    if config_instance.rl_model_name in algo_para_rm_from_base:
        for rm_field in algo_para_rm_from_base[config_instance.rl_model_name]:
            if rm_field in config_instance.model_para:
                del config_instance.model_para[rm_field] 