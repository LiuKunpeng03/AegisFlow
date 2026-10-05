import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]


plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
]
base = {

    'diffusion': {
        ## model
        'model': 'models.TemporalUnet', # TemporalUnet, ConditionalUnet1D (Not implemented yet)
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 2e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 50000,
        'sample_freq': 50000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda:0',
    },

    'plan': {
        'batch_size': 1,
        'device': 'cuda:0',

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': 'test',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

}

cfm = {

    'diffusion': {
        ## model
        'model': 'models.TemporalUnet', # TemporalUnet, ConditionalUnet1D (Not implemented yet)
        'diffusion': 'models.CFM',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'cfm/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 5000, # base: 10000
        'loss_type': 'l2',
        'n_train_steps': 1e6, # base: 2e6
        'batch_size': 64,     # base: 32
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 1, # base: 2
        'ema_decay': 0.995,
        'save_freq': 50000,
        'sample_freq': 50000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda:0',
    },

    'plan': {
        'batch_size': 1,
        'device': 'cuda:0',

        ## cfm model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': 'test',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'f:cfm/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

}

#------------------------ overrides ------------------------#

'''
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
'''

maze2d_umaze_v1 = {
    'diffusion': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
    'plan': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
}

maze2d_large_v1 = {
    'diffusion': {
        'horizon': 384,
        'n_diffusion_steps': 256,  # 256 debug
    },
    'plan': {
        'horizon': 384,
        'n_diffusion_steps': 256,  # 256 debug
        'device': 'cuda:0',
        # Safety enabled
        'safety_enabled': True,  # True, False 
        # (If you want to use naive FM, you also need to change conditional_sample in cfm.py)

        # AegisFlow guidance (inference-time)
        'aegis_enabled': True,
        'aegis_global_em': True,
        'aegis_use_cbf_projection': True,
        'aegis_iterations': 10,
        'aegis_lr': 0.10,
        'aegis_decay': 0.0,
        'aegis_em_rollout_steps': 4,
        'aegis_lambda': 1.0,
        'aegis_beta': 0.10,
        'aegis_barrier_margin': 0.05,
        'aegis_barrier_temp': 0.10,
        'aegis_barrier_delta': 0.05,
        'aegis_barrier_denom_floor': 1e-4,
        'aegis_barrier_value_clip': 100.0,
        'aegis_barrier_grad_clip': 100.0,
        'aegis_barrier_time_power': 1.0,
        'aegis_reward_scale': 1.0,
        'aegis_goal_radius': 0.08,
        'aegis_score_goal_threshold': 0.5,
        'aegis_score_horizon': -1,
        'aegis_early_weight': 0.8,
        'aegis_terminal_weight': 0.0,
        # Smoothness-Psi knobs (fallback to early/terminal weights if omitted).
        'aegis_curvature_weight': 0.,
        'aegis_acceleration_weight': 1.,
        'aegis_smooth_dt': 1e-2,
        'aegis_preserve_best': True,
        'aegis_best_metric': 'real_score',
        'aegis_best_barrier_weight': 0.10,

        # One Shot Initialization
        'one_shot_enabled': True,  # True, False

        ## CBF for flow matching
        'obstacles': [
            {'order': 2, 'center': (5.6, 4.8), 'radius': 1},
            {'order': 4, 'center': (5.1, 1.8), 'radius': 1},
        ],
        # Original:
        # 'obstacles': [
        #     {'order': 2, 'center': (5.6, 4.8), 'radius': 1},
        #     {'order': 4, 'center': (5.1, 1.8), 'radius': 1},
        # ],
        #     'obstacles': [
        #     {'order': 2, 'center': (5.8, 5.0), 'radius': 1},
        #     {'order': 4, 'center': (5.3, 2.0), 'radius': 1},
        # ],
        'cbf_solver': 'closed_form',   # 'qp', 'closed_form'
        'cbf_method': 'relax', # 'robust', 'relax', 'time'
        'robust_term': 0.01,  # 0.01, 0.1
        'relax_threshold': 0.99, # 0.9999
        "eps": .5,
        "rho": .9,

        # set suffix
        'suffix': 'test'
    },
}
