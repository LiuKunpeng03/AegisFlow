import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import torch
from torch import nn
import torchdiffeq
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.core import NeuralODE
from torch.distributions.normal import Normal
import diffuser.utils as utils
import pdb
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

# For NLL
class SimpleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # expects model(x, cond, t)

    def forward(self, x, t):
        t_batch = torch.full((x.shape[0],), t, device=x.device)
        return self.model(x, None, t_batch)

class CFM(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        # CFM setting
        sigma = 0.0
        self.FM = ConditionalFlowMatcher(sigma=sigma)
        self.node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

        # Get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

        # One-shot initialization
        self.one_shot_enabled = False

        # Safety
        self.safety_enabled = False
        self.cbf = None
        self.norm_mins = 0
        self.norm_maxs = 0
        self.safe1 = 0
        self.safe2 = 0

        # Aegis guidance (optional, inference-time)
        self.aegis_enabled = True
        self.aegis_use_cbf_projection = True
        self.aegis = None
        self.latest_aegis_stats = {}

        # Settings for compatibility with diffusion models (Not important for CFM)
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        # manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#
    @torch.no_grad()
    def conditioned_ode_func_record(self, t, x, cond, trajectory_list):
        """
        ODE vector field function with conditioning applied at each step.
        
        t (float or tensor): Current time step in the ODE solver.
        x (tensor): Current state.
        cond (dict): Conditioning dictionary used to fix specific time steps or values.
        trajectory_list (list): List to accumulate the trajectory.
        
        vt (tensor): Vector field output from the model.
        """
        trajectory_list.append(x)  # Append the current state to the trajectory list

        # 1. Apply conditioning to the current state
        x_cond = apply_conditioning(x, cond, self.action_dim)
        
        # 2. Compute the vector field from the conditioned state
        t_batch = torch.full((x.shape[0],), t, device=x.device)
        vt = self.model(x_cond, None, t_batch)

        return vt

    @torch.no_grad()
    def conditioned_ode_func(self, t, x, cond):
        """
        Computes the ODE vector field with conditioning at each step
        """
        # 1. Apply condition to current state
        x_cond = apply_conditioning(x, cond, self.action_dim)
        
        # 2. Compute vector field on the conditioned state
        t_batch = torch.full((x.shape[0],), t, device=x.device)
        vt = self.model(x_cond, None, t_batch)
        
        return vt

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, record_traj=False):
        """
        Generate samples by solving the conditional ODE
        """
        # Initial noise
        x0 = torch.randn(shape).to(self.device)
        
        # Apply condition to initial state
        x0 = apply_conditioning(x0, cond, self.action_dim)

        # Wrapper function for torchdiffeq.odeint (must accept only t and x as arguments)
        if record_traj:
            trajectory_list = []
            ode_fn = lambda t, x: self.conditioned_ode_func_record(t, x, cond, trajectory_list)
        else:
            ode_fn = lambda t, x: self.conditioned_ode_func(t, x, cond)
        
        iter_start = time.time()
        # Solve ODE using wrapper
        traj = torchdiffeq.odeint(
            ode_fn,
            x0,
            torch.linspace(0, 1, self.n_timesteps + 1).to(self.device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        iter_end = time.time()
        iter_time = iter_end - iter_start

        x1 = traj[-1]
        # Apply condition again at the end (for safety)
        x1 = apply_conditioning(x1, cond, self.action_dim)
        
        # pdb.set_trace()
        if record_traj:
            trajectory_list.append(x1) # append last step x
            return x1, torch.stack(trajectory_list, dim=1), [iter_time/self.n_timesteps]
        return x1
    
    @torch.no_grad()
    def p_sample_loop_ode_planning(self, shape, cond, verbose=True, record_traj=False):
        """
        Solve ODE planning with explicit control-corrected RHS (e.g., CBF applied)
        """
        self.aegis_use_cbf_projection=True
        # if self.aegis_enabled:
        #     self.aegis_use_cbf_projection=False # Ablation for CBF projection in Aegis
        OSI_start = time.time()
        n_timesteps = self.n_timesteps
        pred_n_timesteps = 1  # number of prediction steps
        # ================ Prediction Stage ================
        if self.one_shot_enabled:
            batch_size = len(cond[0])
            x0_1st_phase = torch.randn(shape).to(self.device)
            x0_1st_phase = apply_conditioning(x0_1st_phase, cond, self.action_dim)
            
            pred_time_list = torch.linspace(0, 1, pred_n_timesteps+1).to(self.device)
            for i in range(pred_n_timesteps):
                t_now = pred_time_list[i]
                dt = 1 / pred_n_timesteps
                v_t = self.model(x0_1st_phase, None, torch.full((batch_size,), t_now, device=x0_1st_phase.device))
                x0_1st_phase = x0_1st_phase + v_t * dt
                x0_1st_phase = apply_conditioning(x0_1st_phase, cond, self.action_dim)
            x0_2nd_phase = x0_1st_phase
        # ================ Correction Stage ================
        else:
            x0_2nd_phase = torch.randn(shape).to(self.device)
        OSI_end = time.time()
        OSI_time = OSI_end - OSI_start

        x0_2nd_phase = apply_conditioning(x0_2nd_phase, cond, self.action_dim)

        T = n_timesteps + 1

        # Adaptive scheduling
        # time_list = self.adaptive_scheduling(T, device=self.device)
        # Uniform scheduling
        time_list = torch.linspace(0, 1, T).to(self.device)  # [0, 1] for uniform scheduling
        # print(f"Adaptive scheduling: {time_list}")

        # Precompute step sizes for planning steps.
        if self.one_shot_enabled:
            z = 2 * (n_timesteps + 1) / n_timesteps
            dt_list = []
            for i in range(1, T):
                one_minus_t = (n_timesteps - (i - 1)) / n_timesteps
                dt_list.append(z * one_minus_t * (1 / n_timesteps))
            dt_list = torch.tensor(dt_list, device=self.device, dtype=time_list.dtype)
        else:
            dt_list = time_list[1:] - time_list[:-1]
        
        traj = [x0_2nd_phase]

        goal_cond = cond.get(self.horizon - 1, None)
        aegis_buffer = {
            'psi': [],
            'barrier': [],
            'target_grad_norm': [],
            'safe_grad_norm': [],
            'control_delta_norm': [],
            'lambda_t': [],
            'beta_t': [],
            's_t': [],
        }

        # Fully joint global EM path for Aegis.
        if self.aegis_enabled and self.aegis is not None and goal_cond is not None and getattr(self.aegis, 'global_em_enabled', True):
            global_start = time.time()

            def _aegis_model_fn(x_state, t_val):
                t_vec = torch.full((x_state.shape[0],), t_val, device=x_state.device)
                return self.model(x_state, None, t_vec)

            def _aegis_condition_fn(x_state):
                return apply_conditioning(x_state, cond, self.action_dim)

            def _aegis_safety_project_fn(x_now, x_next_naive, t_now):
                if self.safety_enabled and self.cbf is not None and self.aegis_use_cbf_projection:
                    x_corr, _ = self.cbf.apply(x_now, x_next_naive, t=t_now)
                    return x_corr
                return x_next_naive

            traj_tensor, global_stats = self.aegis.optimize_global_controls(
                x_init=x0_2nd_phase,
                time_list=time_list,
                dt_list=dt_list,
                goal_obs=goal_cond,
                model_fn=_aegis_model_fn,
                condition_fn=_aegis_condition_fn,
                safety_project_fn=_aegis_safety_project_fn,
            )

            global_time = time.time() - global_start
            self.latest_aegis_stats = dict(global_stats)
            self.latest_aegis_stats['global_em_total_time_sec'] = float(global_time)

            if record_traj:
                return traj_tensor[:, T - 1, :, :], traj_tensor, [global_time / n_timesteps]
            return traj_tensor[:, T - 1, :, :]
        
        iter_time = 0
        z =  2*(n_timesteps+1) / n_timesteps  # for scaling one-shot init velocity
        for i in range(1, T):
            iter_start = time.time()
            # print(f"{i}-th iter / {T} (time: {t_act1 - t_start:.2f}s)", end="\r")
            t_now = time_list[i-1]
            # define dt based on scheduling
            if self.one_shot_enabled:
                # dt = (2/(n_timesteps *(n_timesteps + 1))) * (n_timesteps-(i-1)) 
                dt = dt_list[i - 1]
            else:
                dt = dt_list[i - 1]
            x_now = traj[-1]
            
            B = x_now.shape[0]
            t_batch = torch.full((B,), t_now, device=x_now.device)
            # Step forward via some base policy (e.g., learned dynamics model)
            u_raw = self.model(x_now, None, t_batch)  # [B, H, D] - same shape as dx/dt

            u_plan = u_raw
            # print(f'aegis_enabled: {self.aegis_enabled}')
            if self.aegis_enabled and self.aegis is not None and goal_cond is not None:
                def _aegis_model_fn(x_state, t_val):
                    t_vec = torch.full((x_state.shape[0],), t_val, device=x_state.device)
                    return self.model(x_state, None, t_vec)

                def _aegis_condition_fn(x_state):
                    return apply_conditioning(x_state, cond, self.action_dim)

                u_plan, step_stats = self.aegis.correct_control(
                    x_now=x_now,
                    u_raw=u_raw,
                    dt=dt,
                    t_scalar=float(t_now.item()),
                    goal_obs=goal_cond,
                    model_fn=_aegis_model_fn,
                    condition_fn=_aegis_condition_fn,
                )
                for key in aegis_buffer:
                    if key in step_stats:
                        aegis_buffer[key].append(step_stats[key])

            # CBF correction
            if self.safety_enabled and self.cbf is not None and self.aegis_use_cbf_projection:
                x_next_naive = x_now + u_plan * dt
                x_corr, _ = self.cbf.apply(x_now, x_next_naive, t=t_now)
                dx = x_corr - x_now
            else:
                dx = u_plan * dt

            x_next = x_now + dx
                    
            x_next = apply_conditioning(x_next, cond, self.action_dim)

            traj.append(x_next)
            iter_end = time.time()
            iter_time += (iter_end - iter_start)

        traj_tensor = torch.stack(traj, dim=1)  # [T, B, H, D]

        if self.aegis_enabled:
            stats = {'enabled': True}
            for key, values in aegis_buffer.items():
                stats[key] = float(np.mean(values)) if len(values) > 0 else 0.0
            self.latest_aegis_stats = stats
        else:
            self.latest_aegis_stats = {'enabled': False}

        if record_traj:
            return traj_tensor[:,T-1,:,:], traj_tensor, [iter_time/n_timesteps]  # sample, diffusion_paths, avg_iter_time
        else:
            return traj_tensor[:,T-1,:,:]               # just sample

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, record_traj=True, **kwargs):
        '''
        conditions : [ (time, state), ... ]
        '''
        # device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        
        #if self.safety_enabled: # Planning
        if True: # Planning
            return self.p_sample_loop_ode_planning(shape, cond, record_traj=record_traj, *args, **kwargs)
        else: # Training or Planning without CBF #TODO: separate training and planning
            # return self.p_sample_loop_ode_planning(shape, cond, record_traj=record_traj, *args, **kwargs) # Planning without CBF
            return self.p_sample_loop(shape, cond, record_traj=record_traj, *args, **kwargs) # Training

    @property
    def device(self):
        """
        Get the device where the model's parameters are allocated
        """
        # Assumes the model's parameters are all on the same device.
        return next(self.parameters()).device
    
    #------------------------------------------ training ------------------------------------------#
    
    def loss(self, x, cond):
        x = x.to(self.device)
        batch_size = len(x)

        # t = torch.rand(batch_size, device=x.device)
        
        x1 = x.to(self.device)
        x0 = torch.randn_like(x1)

        # Generate xt and flow field ut at time t
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)

        # Apply condition
        xt = apply_conditioning(xt, cond, self.action_dim)

        # Compute vector field
        vt = self.model(xt, cond, t) # if there are cond, modify None -> cond

        # Compute loss
        loss, info = self.loss_fn(vt, ut)
        
        return loss, info

    # wthout segementing
    def forward(self, cond, n_diffusion_steps, *args, **kwargs): 
        self.n_timesteps = int(n_diffusion_steps)  # sphagetti code but quick fix
        
        x1, traj, iter_per_time =  self.conditional_sample(cond=cond, *args, **kwargs)

        safe_l, cbf_warn = self.cbf.cbf_nv(x1)
        
        return x1, traj, iter_per_time
