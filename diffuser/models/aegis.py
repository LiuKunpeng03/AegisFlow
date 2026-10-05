import torch

class AegisGuidance:
    """
    Inference-time Aegis-style guidance for Maze2d planning.

    The guidance composes:
    1) target term from a differentiable score proxy, and
    2) safety term from a soft-margin barrier around obstacles.
    """

    def __init__(self, norm_mins, norm_maxs, args):
        self.device = norm_mins.device

        self.norm_mins = norm_mins
        self.norm_maxs = norm_maxs
        self.obstacles = getattr(args, 'obstacles', [])
        
        self.action_dim = 2
        pos_indices = getattr(args, 'cbf_pos_indices', None)
        if pos_indices is None:
            self.pos_y_idx = self.action_dim
            self.pos_x_idx = self.action_dim + 1
        else:
            self.pos_x_idx = self.action_dim + int(pos_indices.get('x', 0))
            self.pos_y_idx = self.action_dim + int(pos_indices.get('y', 1))

        # Aegis runtime knobs
        self.enabled = getattr(args, 'aegis_enabled', False)
        self.global_em_enabled = bool(getattr(args, 'aegis_global_em', True))
        self.iterations = int(getattr(args, 'aegis_iterations', 1))
        self.lr = float(getattr(args, 'aegis_lr', 0.1))
        self.decay = float(getattr(args, 'aegis_decay', 0.0))
        self.em_rollout_steps = int(getattr(args, 'aegis_em_rollout_steps', 4))
        self.preserve_best = bool(getattr(args, 'aegis_preserve_best', True))
        self.best_metric = str(getattr(args, 'aegis_best_metric', 'real_score')).lower()

        # Constant control weights.
        # Backward-compatible fallback keeps old start/end keys usable.
        self.lambda_const = float(
            getattr(args, 'aegis_lambda', getattr(args, 'aegis_lambda_start', 1.0))
        )
        self.beta_const = float(
            getattr(args, 'aegis_beta', getattr(args, 'aegis_beta_start', 0.1))
        )
        self.best_barrier_weight = float(getattr(args, 'aegis_best_barrier_weight', self.beta_const))

        # Barrier / reward proxy settings
        self.barrier_margin = float(getattr(args, 'aegis_barrier_margin', 0.05))
        self.barrier_temp = float(getattr(args, 'aegis_barrier_temp', 0.1))
        self.barrier_delta = float(getattr(args, 'aegis_barrier_delta', 0.05))
        self.barrier_denom_floor = float(getattr(args, 'aegis_barrier_denom_floor', 1e-4))
        self.barrier_value_clip = float(getattr(args, 'aegis_barrier_value_clip', 100.0))
        self.barrier_grad_clip = float(getattr(args, 'aegis_barrier_grad_clip', 100.0))
        self.barrier_time_power = float(getattr(args, 'aegis_barrier_time_power', 1.0))
        self.reward_scale = float(getattr(args, 'aegis_reward_scale', 1.0))
        self.goal_radius = float(getattr(args, 'aegis_goal_radius', 0.08))
        self.score_goal_threshold = float(getattr(args, 'aegis_score_goal_threshold', 0.5))
        self.score_horizon = int(getattr(args, 'aegis_score_horizon', -1))

        self.early_weight = float(getattr(args, 'aegis_early_weight', 0.8))
        self.terminal_weight = float(getattr(args, 'aegis_terminal_weight', 0.2))
        self.curvature_weight = float(getattr(args, 'aegis_curvature_weight', self.early_weight))
        self.acceleration_weight = float(getattr(args, 'aegis_acceleration_weight', self.terminal_weight))
        self.smooth_dt = float(getattr(args, 'aegis_smooth_dt', 1e-2))

        print(f'reward_scale: {self.reward_scale}, curvature_weight: {self.curvature_weight}, acceleration_weight: {self.acceleration_weight}')

        # Precompute normalization factors exactly as in CBF code.
        self.xr = 2 / (self.norm_maxs[1] - self.norm_mins[1])
        self.yr = 2 / (self.norm_maxs[0] - self.norm_mins[0])

    def _compute_s(self, t_scalar):
        """
        Constant-lambda approximation:
        s(t) = integral_t^1 1/lambda d tau = (1 - t) / lambda.
        """
        lam = max(self.lambda_const, 1e-6)
        return max((1.0 - t_scalar) / lam, 0.0)

    def _compute_target_proxy(self, x_pred, goal_obs):
        """
        Differentiable smoothness objective for Maze2d trajectory states.

        Psi = reward_scale * (w_c * c_smooth + w_a * s_smooth)
        where c_smooth and s_smooth follow the same definitions used in
        testing (curvature and acceleration smoothness).
        """
        del goal_obs  # kept for interface compatibility

        with torch.enable_grad():
            x_var = x_pred.detach().clone().requires_grad_(True)
            c_smooth, s_smooth = self._compute_smooth_metrics(x_var)
            psi = self.reward_scale * (
                self.curvature_weight * c_smooth +
                self.acceleration_weight * s_smooth
            )
            grad = torch.autograd.grad(
                psi.sum(),
                x_var,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )[0]

        return psi.detach(), grad.detach(), c_smooth.detach(), s_smooth.detach()

    def _compute_smooth_metrics(self, x_pred, eps_area: float = 1e-12, eps_len: float = 1e-12, eps_dt: float = 1e-12):
        """
        Differentiable c_smooth and s_smooth that match the test-time definitions.
        Returns tensors of shape [B].
        """
        X = x_pred
        if X.dim() == 2:
            X = X.unsqueeze(0)
        assert X.dim() == 3, "x_pred must be [B, H, D] or [H, D]"

        B, T, _ = X.shape
        if T < 3:
            zeros = X.new_zeros((B,))
            return zeros, zeros

        # Extract coordinates (same indexing as CBF smoothness metrics).
        y = X[:, :, self.pos_y_idx]
        x = X[:, :, self.pos_x_idx]
        P = torch.stack([x, y], dim=-1)  # [B, T, 2]

        # ----- c_smooth (curvature) -----
        p_im1 = P[:, :-2, :]
        p_i = P[:, 1:-1, :]
        p_ip1 = P[:, 2:, :]

        v1 = p_i - p_im1
        v2 = p_ip1 - p_i
        v3 = p_ip1 - p_im1

        a = torch.linalg.norm(v1, dim=-1)
        b = torch.linalg.norm(v2, dim=-1)
        c = torch.linalg.norm(v3, dim=-1)

        cross_z = v1[..., 0] * v3[..., 1] - v1[..., 1] * v3[..., 0]
        area = 0.5 * cross_z.abs()

        denom = (a * b * c).clamp(min=eps_len)
        kappa = (4.0 * area) / denom

        mask_curv = (a > eps_len) & (b > eps_len) & (c > eps_len) & (area > eps_area) & torch.isfinite(kappa)
        kappa = torch.where(mask_curv, kappa, torch.zeros_like(kappa))
        curv_sum = kappa.sum(dim=1)
        curv_count = mask_curv.sum(dim=1).clamp(min=1)
        c_smooth = curv_sum / curv_count

        # ----- s_smooth (acceleration) -----
        second_diff = P[:, 2:, :] - 2.0 * P[:, 1:-1, :] + P[:, :-2, :]
        dt2 = torch.as_tensor(self.smooth_dt, dtype=P.dtype, device=P.device)
        dt2 = (dt2 * dt2).clamp(min=eps_dt * eps_dt)
        a_vec = second_diff / dt2
        a_mag = torch.linalg.norm(a_vec, dim=-1)

        mask_acc = torch.isfinite(a_mag)
        acc_sum = torch.where(mask_acc, a_mag, torch.zeros_like(a_mag)).sum(dim=1)
        acc_count = mask_acc.sum(dim=1).clamp(min=1)
        s_smooth = acc_sum / acc_count

        return c_smooth, s_smooth

    def _compute_soft_barrier(self, x_pred, t_scalar=0.0):
        """
        Time-varying reciprocal barrier around CBF obstacles.

        Safe-set function:
            h_i(x, t) = dy^n + dx^n - t
        Barrier:
            B_i(x, t) = 1 / (h_i(x, t) + delta)
        """
        # is_maze = False
        if not self.obstacles:
            barrier = torch.zeros((x_pred.shape[0],), device=x_pred.device)
            grad = torch.zeros_like(x_pred)
            return barrier, grad

        if isinstance(t_scalar, torch.Tensor):
            t_scalar = float(t_scalar.detach().item())
        t_eff = max(min(float(t_scalar), 1.0), 0.0)
        t_eff = t_eff ** max(self.barrier_time_power, 1e-6)

        denom_floor = max(self.barrier_denom_floor, 1e-8)
        value_clip = max(self.barrier_value_clip, denom_floor)
        grad_clip = max(self.barrier_grad_clip, 1e-8)

        y = x_pred[:, :, self.pos_y_idx]
        x = x_pred[:, :, self.pos_x_idx]
        # print(f'x_pred shape: {x_pred.shape}, y shape: {y.shape}, x shape: {x.shape}')
        grad_y_total = torch.zeros_like(y)
        grad_x_total = torch.zeros_like(x)
        barrier_accum = torch.zeros((x_pred.shape[0],), device=x_pred.device)

        def _accumulate_barrier(barrier_value, grad_y, grad_x):
            nonlocal barrier_accum, grad_y_total, grad_x_total
            barrier_accum += barrier_value
            grad_y_total += grad_y
            grad_x_total += grad_x


        for obs in self.obstacles:
            
            cx, cy = obs['center']
            order = float(obs['order'])
            off_x = 2 * (cx - 0.5 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1
            off_y = 2 * (cy - 0.5 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
            r = float(obs['radius'])
            dx = (x - off_x) / (self.xr * r)
            dy = (y - off_y) / (self.yr * r)
            h = torch.pow(torch.pow(dy, order) + torch.pow(dx, order), 1.0 / order) - t_eff
            h *= r#**order  # scale back to original units for barrier computation
            denom = torch.clamp(h + self.barrier_delta, min=denom_floor)
            b = 1.0 / denom
            b = torch.nan_to_num(b, nan=value_clip, posinf=value_clip, neginf=value_clip)
            b = torch.clamp(b, min=0.0, max=value_clip)
            _accumulate_barrier(b.mean(dim=1) * order / 2, torch.zeros_like(y), torch.zeros_like(x))
            db_dh = -(1.0 / (denom * denom))
            db_dh = torch.nan_to_num(db_dh, nan=0.0, posinf=grad_clip, neginf=-grad_clip)
            db_dh = torch.clamp(db_dh, min=-grad_clip, max=grad_clip)
            dh_dy = order * torch.pow(dy, order - 1.0) / self.yr
            dh_dx = order * torch.pow(dx, order - 1.0) / self.xr
            grad_y_total += db_dh * dh_dy * order/2 
            grad_x_total += db_dh * dh_dx * order/2 

        grad = torch.zeros_like(x_pred)
        grad[:, :, self.pos_y_idx] = grad_y_total
        grad[:, :, self.pos_x_idx] = grad_x_total
        return barrier_accum, grad

    def _rollout_controlled_flow(self, x_now, t_scalar, delta, model_fn, condition_fn=None):
        """
        E-step style rollout using the controlled reference flow b* = b + u.
        """
        if model_fn is None:
            return x_now

        n_steps = max(self.em_rollout_steps, 1)
        remaining = max(1.0 - float(t_scalar), 0.0)
        if remaining <= 1e-8:
            return x_now

        dt_inner = remaining / float(n_steps)
        x_roll = x_now
        for step_idx in range(n_steps):
            t_inner = float(t_scalar) + step_idx * dt_inner
            u_base = model_fn(x_roll, t_inner)
            x_roll = x_roll + (u_base + delta) * dt_inner
            if condition_fn is not None:
                x_roll = condition_fn(x_roll)
        return x_roll

    def _scalar(self, value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)

    def _select_score(self, psi_mean, barrier_mean, real_score=None):
        if self.best_metric == 'real_score' and real_score is not None:
            return float(real_score)
        if self.best_metric == 'psi':
            return float(psi_mean)
        return float(psi_mean - self.best_barrier_weight * barrier_mean)

    @torch.no_grad()
    def _compute_real_score(self, x_state, goal_obs):
        """
        Maze2d-style score from a planned trajectory state.

        We estimate sparse return as the number of remaining steps after first
        entry into a goal neighborhood, then normalize by horizon.
        """
        pos = x_state[:, :, self.pos_y_idx:self.pos_x_idx + 1]  # [B, H, 2]
        goal_pos = goal_obs[:, :2].unsqueeze(1)  # [B, 1, 2]

        # Align with environment stepping: rewards are from executed steps,
        # so ignore the initial waypoint at index 0.
        pos_step = pos[:, 1:, :]
        dist = torch.norm(pos_step - goal_pos, dim=-1)  # [B, H-1]
        hit_mask = dist <= self.score_goal_threshold

        n_step = dist.shape[1]
        any_hit = hit_mask.any(dim=1)
        first_hit = hit_mask.float().argmax(dim=1)

        remaining_reward = torch.where(
            any_hit,
            (n_step - first_hit).to(dist.dtype),
            torch.zeros_like(first_hit, dtype=dist.dtype),
        )

        horizon = self.score_horizon if self.score_horizon > 0 else n_step
        horizon = max(int(horizon), 1)
        real_score = remaining_reward / float(horizon)
        return float(real_score.mean().detach().cpu().item())

    @torch.no_grad()
    def _trajectory_metrics(self, traj, goal_obs, time_list=None):
        """
        Compute selection metrics on a rolled-out trajectory.
        """
        terminal_state = traj[:, -1]
        psi_terminal, _, _, _ = self._compute_target_proxy(terminal_state, goal_obs)
        psi_mean = float(psi_terminal.mean().detach().cpu().item())
        real_score = self._compute_real_score(terminal_state, goal_obs)

        n_steps = traj.shape[1] - 1
        barrier_vals = []
        for j in range(n_steps):
            if time_list is not None and j < len(time_list):
                t_j = self._scalar(time_list[j])
            else:
                t_j = float(j) / max(float(n_steps), 1.0)
            barrier_val, _ = self._compute_soft_barrier(traj[:, j], t_scalar=t_j)
            barrier_vals.append(float(barrier_val.mean().detach().cpu().item()))
        barrier_mean = float(sum(barrier_vals) / max(len(barrier_vals), 1))

        score = self._select_score(psi_mean, barrier_mean, real_score=real_score)
        return psi_mean, barrier_mean, real_score, score

    @torch.no_grad()
    def _rollout_with_control_map(self, x_init, time_list, dt_list, control_map, model_fn, condition_fn=None, safety_project_fn=None):
        """
        Roll out full dynamics with a global control map u_0...u_{N-1}.
        """
        traj = [x_init]
        for step_idx in range(len(control_map)):
            x_now = traj[-1]
            t_now = self._scalar(time_list[step_idx])
            dt = self._scalar(dt_list[step_idx])

            drift = model_fn(x_now, t_now)
            x_next_naive = x_now + (drift + control_map[step_idx]) * dt

            if safety_project_fn is not None:
                x_next = safety_project_fn(x_now, x_next_naive, t_now)
            else:
                x_next = x_next_naive

            if condition_fn is not None:
                x_next = condition_fn(x_next)

            traj.append(x_next)

        return torch.stack(traj, dim=1)

    def optimize_global_controls(
        self,
        x_init,
        time_list,
        dt_list,
        goal_obs,
        model_fn,
        condition_fn=None,
        safety_project_fn=None,
    ):
        """
        Fully joint global EM-style optimization over the control sequence.

        E-step: rollout with current b* = b + u map.
        M-step: update all u_n jointly using terminal objective pullback + barrier gradients.
        """
        if (not self.enabled) or (goal_obs is None):
            zero_map = [torch.zeros_like(x_init) for _ in range(len(dt_list))]
            traj = self._rollout_with_control_map(
                x_init=x_init,
                time_list=time_list,
                dt_list=dt_list,
                control_map=zero_map,
                model_fn=model_fn,
                condition_fn=condition_fn,
                safety_project_fn=safety_project_fn,
            )
            return traj, {'enabled': False, 'mode': 'off'}

        n_steps = len(dt_list)
        control_map = [torch.zeros_like(x_init) for _ in range(n_steps)]

        lambda_t = max(self.lambda_const, 1e-6)
        beta_t = self.beta_const

        best_score = float('-inf')
        best_iter = -1
        best_psi = float('inf')
        best_barrier = 0.0
        best_real_score = 0.0
        best_control_map = [u.detach().clone() for u in control_map]

        for i in range(max(self.iterations, 1)):
            iter_control_map = [u.detach().clone() for u in control_map]

            # E-step: rollout full trajectory with current controlled reference flow b*=b+u.
            traj = self._rollout_with_control_map(
                x_init=x_init,
                time_list=time_list,
                dt_list=dt_list,
                control_map=control_map,
                model_fn=model_fn,
                condition_fn=condition_fn,
                # safety_project_fn=safety_project_fn,
            )

            # Terminal objective gradient from Psi directly.
            # For smoothness Psi (cost), initialize co-state with lambda_N = grad_x(Psi).
            x_terminal = traj[:, -1]
            psi_terminal, g_terminal, c_smooth, s_smooth = self._compute_target_proxy(x_terminal, goal_obs)
            iter_psi = float(psi_terminal.mean().detach().cpu().item())
            if iter_psi < best_psi:
                print(f"New best psi found at iteration {i}: {iter_psi:.4f} (previous best: {best_psi:.4f})")
                best_psi = iter_psi
                best_iter = i
                best_control_map = [u.detach().clone() for u in iter_control_map]
            lam = g_terminal.detach()

            per_step_tar = [None for _ in range(n_steps)]
            per_step_safe = [None for _ in range(n_steps)]
            per_step_barrier = []

            # M-step (global): pull terminal gradient backward through all controlled steps.
            with torch.enable_grad():
                for j in range(n_steps - 1, -1, -1):
                    x_j = traj[:, j].detach().clone()
                    u_j = control_map[j].detach().clone()
                    t_now = self._scalar(time_list[j])
                    dt = self._scalar(dt_list[j])

                    x_in = x_j.reshape(-1).requires_grad_(True)

                    def step_func(x_flat):
                        x_state = x_flat.contiguous().reshape_as(x_j)
                        drift = model_fn(x_state, t_now)
                        x_next = x_state + (drift + u_j) * dt
                        if condition_fn is not None:
                            x_next = condition_fn(x_next)
                        return x_next.reshape(-1)

                    _, vjp = torch.autograd.functional.vjp(
                        step_func,
                        inputs=x_in,
                        v=lam.reshape(-1),
                        create_graph=False,
                        strict=False,
                    )
                    lam = vjp.reshape_as(x_j).detach()
                    per_step_tar[j] = lam

                    barrier_val, g_safe = self._compute_soft_barrier(x_j, t_scalar=t_now)
                    per_step_safe[j] = g_safe.detach()
                    per_step_barrier.append(float(barrier_val.mean().detach().cpu().item()))

            # Joint control-map update.
            for j in range(n_steps):
                t_frac = self._scalar(time_list[j])
                s_t = self._compute_s(t_frac)
                update = -(1.0 / lambda_t) * per_step_tar[j] - (s_t * beta_t) * per_step_safe[j]
                control_map[j] = self.decay * control_map[j] + self.lr * update

            print(f"Aegis iteration {i}: psi: {psi_terminal.item():.4f}, barrier: {per_step_barrier[0]:.4f}, c_smooth: {c_smooth.item():.4f}, s_smooth: {s_smooth.item():.4f}")
    
        # Evaluate the final updated iterate and preserve the best overall candidate.
        traj_last = self._rollout_with_control_map(
            x_init=x_init,
            time_list=time_list,
            dt_list=dt_list,
            control_map=control_map,
            model_fn=model_fn,
            condition_fn=condition_fn,
            safety_project_fn=safety_project_fn,
        )

        x_terminal = traj_last[:, -1]
        psi_terminal, g_terminal, c_smooth, s_smooth = self._compute_target_proxy(x_terminal, goal_obs)
        last_psi = float(psi_terminal.mean().detach().cpu().item())

        if last_psi < best_psi:
            print(f"New best psi found at last iteration: {last_psi:.4f} (previous best: {best_psi:.4f})")

        traj_final = traj_last # using last iter
        stats = {}

        return traj_final, stats

    @torch.no_grad()
    def correct_control(self, x_now, u_raw, dt, t_scalar, goal_obs, model_fn=None, condition_fn=None):
        if (not self.enabled) or (goal_obs is None):
            return u_raw, {}

        delta = torch.zeros_like(u_raw)

        best_iter_idx = -1
        best_psi = float('inf')
        best_delta = delta.detach().clone()
        best_barrier = None
        best_g_tar = None
        best_g_safe = None

        lambda_t = max(self.lambda_const, 1e-6)
        beta_t = self.beta_const
        s_t = self._compute_s(t_scalar)

        # EM-like update:
        # E-step: rollout x under b* = b + u
        # M-step: update u from target/safety gradients
        for i in range(max(self.iterations, 1)):
            if model_fn is None:
                x_ref = x_now + (u_raw + delta) * dt
            else:
                x_ref = self._rollout_controlled_flow(
                    x_now=x_now,
                    t_scalar=t_scalar,
                    delta=delta,
                    model_fn=model_fn,
                    condition_fn=condition_fn,
                )

            psi, g_tar, _, _ = self._compute_target_proxy(x_ref, goal_obs)
            barrier, g_safe = self._compute_soft_barrier(x_now, t_scalar=t_scalar)

            psi_mean = float(psi.mean().detach().cpu().item())
            if psi_mean < best_psi:
                best_psi = psi_mean
                best_iter_idx = i
                best_delta = delta.detach().clone()
                best_barrier = barrier.detach().clone()
                best_g_tar = g_tar.detach().clone()
                best_g_safe = g_safe.detach().clone()

            # u* ~ -lambda^{-1} grad(Psi) - s*beta*grad(B)
            correction = -(1.0 / lambda_t) * g_tar - (s_t * beta_t) * g_safe
            delta = self.decay * delta + self.lr * correction

        u_corr = u_raw + best_delta

        stats = {
            'psi': float(best_psi),
            'barrier': float(best_barrier.mean().detach().cpu().item()) if best_barrier is not None else 0.0,
            'target_grad_norm': float(best_g_tar.norm(dim=-1).mean().detach().cpu().item()) if best_g_tar is not None else 0.0,
            'safe_grad_norm': float(best_g_safe.norm(dim=-1).mean().detach().cpu().item()) if best_g_safe is not None else 0.0,
            'control_delta_norm': float(best_delta.norm(dim=-1).mean().detach().cpu().item()),
            'lambda_t': float(lambda_t),
            'beta_t': float(beta_t),
            'barrier_delta': float(self.barrier_delta),
            's_t': float(s_t),
            'em_rollout_steps': float(max(self.em_rollout_steps, 1)),
            'best_iteration': float(best_iter_idx),
        }
        return u_corr, stats