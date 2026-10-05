import json
import numpy as np
from os.path import join, dirname
import pdb
import csv
import time
import os

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import torch

eval_coord_shift = 0.0  # Shift applied to trajectory coordinates for safety evaluation only (not for plotting)

# fix seed
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# fix seed

# diffusion model:
# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --logbase logs --method base
# cfm model:
# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --logbase logs --method cfm
# cfm model with safety disabled:
# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --logbase logs --method cfm --safety false
# cfm model with safety enabled:
# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --logbase logs --method cfm --safety true

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'
    method: str = 'cfm'
    safety: str = None  # Override safety_enabled: 'true', 'false', or None (use config)
    aegis: str = None  # Override aegis_enabled: 'true', 'false', or None (use config)


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# Override safety_enabled if --safety flag is provided
if args.safety is not None:
    args.safety_enabled = args.safety.lower() == 'true'
    print(f"[Override] safety_enabled = {args.safety_enabled}")

# Override aegis_enabled if --aegis flag is provided
if args.aegis is not None:
    args.aegis_enabled = args.aegis.lower() == 'true'
    print(f"[Override] aegis_enabled = {args.aegis_enabled}")

utils.set_device(args.device)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch, device=args.device
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer, args)

# Use environment horizon for Aegis real-score normalization unless explicitly set.
if hasattr(policy.diffusion_model, 'aegis') and policy.diffusion_model.aegis is not None:
    if getattr(policy.diffusion_model.aegis, 'score_horizon', -1) <= 0:
        env_horizon = int(getattr(env, 'max_episode_steps', 0))
        if env_horizon > 0:
            policy.diffusion_model.aegis.score_horizon = env_horizon

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def smooth(diffusion):
    steps, horizon = diffusion.shape[0], diffusion.shape[1]
    diffusion_copy = diffusion.copy()
    for i in range(steps - 20, steps, 1):
        for j in range(5, horizon, 1):
            diffusion_copy[i,j,0:2] = np.mean(diffusion[i, j-5:j, 0:2], axis=0)
    
    return diffusion_copy


def save_flow_trajectory_snapshots(renderer, save_dir, iter_idx, diffusion_paths, time_fracs=None):
    """
    Save full generated trajectories at selected diffusion-time indices.

    diffusion_paths should have shape [n_diffusion_steps, horizon, dim], where
    each diffusion step corresponds to one full trajectory.
    """
    if time_fracs is None:
        time_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]

    if diffusion_paths is None:
        return

    if torch.is_tensor(diffusion_paths):
        diffusion_paths = diffusion_paths.detach().cpu().numpy()
    else:
        diffusion_paths = np.asarray(diffusion_paths)

    if diffusion_paths.ndim != 3:
        return

    n_steps = int(diffusion_paths.shape[0])
    if n_steps <= 0:
        return

    snapshots_dir = join(save_dir, 'results', f'{iter_idx}_flow_snapshots')
    os.makedirs(snapshots_dir, exist_ok=True)

    for frac in time_fracs:
        frac_clamped = float(np.clip(frac, 0.0, 1.0))
        step_idx = int(round(frac_clamped * (n_steps - 1)))
        step_idx = max(0, min(step_idx, n_steps - 1))

        full_traj = diffusion_paths[step_idx:step_idx + 1]
        frac_tag = f'{frac_clamped:.2f}'.replace('.', 'p')
        out_path = join(snapshots_dir, f'flow_t{frac_tag}_step{step_idx:03d}.png')
        renderer.composite(out_path, full_traj, ncol=1)

# --------------------------------- csv header ----------------------------------#
# save results to csv file
csv_path = join(args.savepath, 'results.csv')
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['iter', 'safe1', 'safe2', 'trap1', 'trap2', 'score', 'itertime', 'success'])

#---------------------------------- helper functions ----------------------------------#
def check_position_safety(pos, obstacles, norm_mins, norm_maxs):
    """
    Check if a position is safe (outside all obstacles).
    Returns True if safe, False if inside any obstacle.
    pos: (y, x) position in original coordinates
    """
    yr = 2 / (norm_maxs[0] - norm_mins[0])
    xr = 2 / (norm_maxs[1] - norm_mins[1])

    for obs in obstacles:
        cx, cy = obs['center']
        order = obs['order']

        off_x = 2 * (cx - 0.5 - norm_mins[1]) / (norm_maxs[1] - norm_mins[1]) - 1
        off_y = 2 * (cy - 0.5 - norm_mins[0]) / (norm_maxs[0] - norm_mins[0]) - 1

        # Normalize position
        pos_y_norm = 2 * (pos[0] - 0.5 - norm_mins[0]) / (norm_maxs[0] - norm_mins[0]) - 1
        pos_x_norm = 2 * (pos[1] - 0.5 - norm_mins[1]) / (norm_maxs[1] - norm_mins[1]) - 1

        dx = (pos_x_norm - off_x) / xr
        dy = (pos_y_norm - off_y) / yr

        # CBF value: positive means safe (outside obstacle)
        cbf_value = dy**order + dx**order - 1

        if cbf_value < 0:
            return False  # Inside this obstacle

    return True  # Outside all obstacles


def rollout_safety_details(rollout_arr, obstacles, norm_mins, norm_maxs, coord_shift=0.0):
    """
    Compute obstacle-wise rollout safety details in the exact same frame as cbf_nv.
    Returns one dict per obstacle with min CBF value, argmin timestep and position.
    """
    yr = 2 / (norm_maxs[0] - norm_mins[0])
    xr = 2 / (norm_maxs[1] - norm_mins[1])

    coord_shift = .5

    # rollout_arr format follows the existing script convention: [T, 4] = (y, x, vy, vx)
    # Apply shift for safety evaluation only; keep raw coordinates for plotting.
    eval_y = rollout_arr[:, 0] + coord_shift
    eval_x = rollout_arr[:, 1] + coord_shift
    pos_y_norm = 2 * (eval_y - 0.5 - norm_mins[0]) / (norm_maxs[0] - norm_mins[0]) - 1
    pos_x_norm = 2 * (eval_x - 0.5 - norm_mins[1]) / (norm_maxs[1] - norm_mins[1]) - 1

    details = []
    for obs in obstacles:
        cx, cy = obs['center']
        order = obs['order']

        off_x = 2 * (cx - 0.5 - norm_mins[1]) / (norm_maxs[1] - norm_mins[1]) - 1
        off_y = 2 * (cy - 0.5 - norm_mins[0]) / (norm_maxs[0] - norm_mins[0]) - 1

        dx = (pos_x_norm - off_x) / xr
        dy = (pos_y_norm - off_y) / yr
        cbf_series = dy**order + dx**order - 1

        min_idx = int(np.argmin(cbf_series))
        details.append({
            'order': order,
            'center_x': float(cx),
            'center_y': float(cy),
            'min_cbf': float(cbf_series[min_idx]),
            'min_t': min_idx,
            'min_pos_y': float(rollout_arr[min_idx, 0]),
            'min_pos_x': float(rollout_arr[min_idx, 1]),
            'num_violating': int(np.sum(cbf_series < 0.0)),
        })

    return details

#---------------------------------- main loop ----------------------------------#
safe1_batch, safe2_batch = [], []
safe_detail_batches = [[], []]
score_batch = []
elbo_batch = []
is_trap1, is_trap2 = 0, 0
num_trap1, num_trap2 = 0, 0
c_smooth_batch, s_smooth_batch = [], []
aegis_psi_batch, aegis_barrier_batch = [], []
aegis_delta_batch = []
num_success = 0
num_safe_total = 0  # Counter for trials where both safe1 and safe2 are positive
iter_time_batch = []
plan_total_time_batch = []  # Track planning time only (not rollout)
is_save_vid = False  # if you want to save diffusion process video
is_save_img = True   # if you want to save result trajectories
is_save_rollout = False  # if you want to save rollout image
is_save_img_group = False  # if you want to save diffusion process images
set_cond = False

# Get normalization parameters for safety check
norm_mins = dataset.normalizer.normalizers['observations'].mins
norm_maxs = dataset.normalizer.normalizers['observations'].maxs


# print(f"Obstacles: {obstacles}")
# raise Exception

TOTAL_TEST_ITER = 30
original_obstacles = getattr(args, 'obstacles', [])

for it in range(1, TOTAL_TEST_ITER+1):   # num of testing runs
    print("Total test iteration: ", it, f"/{TOTAL_TEST_ITER}")

    args.obstacles = [
        {'order': 2, 'center': (5.6, 4.8), 'radius': 1},
        {'order': 4, 'center': (5.1, 1.8), 'radius': 1},
    ]
    
    policy = Policy(diffusion, dataset.normalizer, args)

    # Use environment horizon for Aegis real-score normalization unless explicitly set.
    if hasattr(policy.diffusion_model, 'aegis') and policy.diffusion_model.aegis is not None:
        if getattr(policy.diffusion_model.aegis, 'score_horizon', -1) <= 0:
            env_horizon = int(getattr(env, 'max_episode_steps', 0))
            if env_horizon > 0:
                policy.diffusion_model.aegis.score_horizon = env_horizon

    # Random start-goal with safety check
    max_resample_attempts = 100
    resample_count = 0

    observation = env.reset()    #array([ 0.94875744,  8.93648809, -0.01347715,  0.06358764])
    observation = np.array([ 0.94875744,  2.93648809, -0.01347715,  0.06358764])   # fix the initial position and final destination for comparison (not needed for general testing)
    env.set_state(observation[0:2], observation[2:4]) ############################################################ same as the last line

    while True:
        if set_cond:
            # Get random start position
            observation = env.reset()
            # Set random target (goal)
            env.set_target()
            target = env._target

        # If safety is enabled, check if start and goal are safe
        if set_cond and args.safety_enabled and hasattr(args, 'obstacles') and args.obstacles:
            start_pos = observation[:2]  # (y, x)
            goal_pos = target  # (y, x)

            start_safe = check_position_safety(start_pos, args.obstacles, norm_mins, norm_maxs)
            goal_safe = check_position_safety(goal_pos, args.obstacles, norm_mins, norm_maxs)

            if start_safe and goal_safe:
                break  # Both positions are safe
            else:
                resample_count += 1
                if resample_count >= max_resample_attempts:
                    print(f"Warning: Could not find safe start-goal pair after {max_resample_attempts} attempts. Using current positions.")
                    break
        else:
            break  # No safety check needed

    if resample_count > 0:
        print(f"  Resampled {resample_count} times to find safe start-goal pair")

    ## set conditioning xy position to be the goal
    target = env._target
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    env_step = env.max_episode_steps
    # for t in range(env_step):
    for t in range(800): #real is 384

        state = env.state_vector().copy()

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:

            cond[0] = observation
            # Measure planning time only
            plan_start_time = time.time()
            action, samples, diffusion_paths, plan_safe1, plan_safe2, elbo, num_trap, iter_time, c_smooth, s_smooth = policy(cond, batch_size=args.batch_size)
            plan_total_time = time.time() - plan_start_time

            aegis_stats = getattr(policy.diffusion_model, 'latest_aegis_stats', {})
            if aegis_stats.get('enabled', False):
                aegis_psi_batch.append(float(aegis_stats.get('psi', 0.0)))
                aegis_barrier_batch.append(float(aegis_stats.get('barrier', 0.0)))
                aegis_delta_batch.append(float(aegis_stats.get('control_delta_norm', 0.0)))

            elbo_batch.append(elbo)
            # Note: plan_safe1, plan_safe2 are based on planned trajectory (not used for statistics)
            
            actions = samples.actions[0]
            sequence = samples.observations[0]
            velocities = (sequence[1:, :2] - sequence[:-1, :2]) # derive velocity from consecutive positions
            diffusion_paths = diffusion_paths[0]
            disp_mag = np.linalg.norm(velocities, axis=1)   # shape: (383,)
            ##################################################start saving videos/images
            # Save the composite image of all trajectories
            if is_save_img: # if you want to save result trajectories
                fullpath = join(args.savepath, f'results/{it}.png')
                os.makedirs(dirname(fullpath), exist_ok=True)
                renderer.composite(fullpath, samples.observations, ncol=1)
                
                # Save the next step composite image of all trajectoiries
                """
                for checking velocity of each way-points, 
                """
                
            # Save the diffusion process as a video
            if is_save_vid: # if you want to save diffusion process video
                renderer.render_diffusion(join(args.savepath, f'diffusion_{iter}.mp4'), diffusion_paths)
            
            # Save individual frames of the diffusion process
            if is_save_img_group: # if you want to save diffusion process images
                diff_step = diffusion_paths.shape[0]  
                makedirs(join(args.savepath, 'png'))
                for kk in range(diff_step):
                    imgpath = join(args.savepath, f'png/{kk}.png')
                    renderer.composite(imgpath, diffusion_paths[kk:kk+1], ncol=1)
            
            #################################################end saving videos/images

        ##################################################start planning
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            
        # can use actions or define a simple controller based on state predictions
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        if t==0:
            prev_action = action.copy()
            prev_state = state.copy()
        prev_action = action.copy()
        prev_state = state.copy()

        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        ##################################################end planning

        ##################################################start logging
    
        ## update rollout observations
        rollout.append(next_observation.copy())
        
        if terminal:
            break

        observation = next_observation
    
    ######################
    rollout_arr = np.array(rollout, copy=True)
    arr = np.expand_dims(rollout_arr, axis=0)
    if is_save_rollout: # if you want to save rollout image
        fullpath = join(args.savepath, f'rollout_{iter}.png')
        renderer.composite(fullpath, arr, ncol=1)

    ##################################################start plan-based safety calculation
    # Compute safety based on planned trajectory (where safety constraints are enforced).
    planned_arr = np.array(sequence, copy=True)  # shape: [T, 4] where 4 = (y, x, vy, vx)
    planned_eval_arr = planned_arr.copy()

    # Normalize planned observations for cbf_nv input.
    planned_normalized = dataset.normalizer.normalize(planned_eval_arr, 'observations')
    # cbf_nv expects [batch, T, action_dim + obs_dim] with action_dim=2.
    dummy_actions = np.zeros((planned_normalized.shape[0], 2))  # [T, 2]
    planned_with_actions = np.concatenate([dummy_actions, planned_normalized], axis=1)  # [T, 6]
    planned_tensor = torch.tensor(planned_with_actions, dtype=torch.float32, device=args.device).unsqueeze(0)

    planned_safe_l = policy.diffusion_model.cbf.cbf_nv(planned_tensor)
    safe1_val, safe2_val = planned_safe_l[0][0], planned_safe_l[0][1]
    planned_details = rollout_safety_details(planned_arr, args.obstacles, norm_mins, norm_maxs, coord_shift=eval_coord_shift)

    # Save planned image with per-obstacle most violating points highlighted.
    if is_save_img and len(planned_details) > 0:
        marker_points = [
            np.array([detail['min_pos_y'], detail['min_pos_x']], dtype=np.float32)
            for detail in planned_details
        ]
        marker_labels = [
            f"obs{obs_idx+1}(p{int(detail['order'])}) min={detail['min_cbf']:.3f}"
            for obs_idx, detail in enumerate(planned_details)
        ]
        diagpath = join(args.savepath, f'results/{it}_plan_violation.png')
        os.makedirs(dirname(diagpath), exist_ok=True)
        renderer.composite(
            diagpath,
            np.expand_dims(planned_arr, axis=0),
            ncol=1,
            violation_point=[marker_points],
            violation_label=[marker_labels],
        )

    # Save executed rollout image with per-obstacle most violating points highlighted.
    # This is for diagnostics only; reported safe statistics remain plan-based.
    rollout_details = rollout_safety_details(rollout_arr, args.obstacles, norm_mins, norm_maxs, coord_shift=eval_coord_shift)
    if is_save_img and len(rollout_details) > 0:
        rollout_marker_points = [
            np.array([detail['min_pos_y'], detail['min_pos_x']], dtype=np.float32)
            for detail in rollout_details
        ]
        rollout_marker_labels = [
            f"obs{obs_idx+1}(p{int(detail['order'])}) min={detail['min_cbf']:.3f}"
            for obs_idx, detail in enumerate(rollout_details)
        ]
        rollout_diagpath = join(args.savepath, f'results/{it}_executed_violation.png')
        os.makedirs(dirname(rollout_diagpath), exist_ok=True)
        renderer.composite(
            rollout_diagpath,
            np.expand_dims(rollout_arr, axis=0),
            ncol=1,
            violation_point=[rollout_marker_points],
            violation_label=[rollout_marker_labels],
        )
    ##################################################end plan-based safety calculation

    ##################################################start statistics calculation
    if reward > 0.95:
        num_success = num_success + 1

    if num_trap >= 1:
        is_trap1 = 1
        num_trap1 += 1
    else:
        is_trap1 = 0
    
    if num_trap >= 2:
        is_trap2 = 1
        num_trap2 += 1
    else:
        is_trap2 = 0

    safe1_batch.append(safe1_val)
    safe2_batch.append(safe2_val)
    for obs_idx, detail in enumerate(planned_details[:2]):
        safe_detail_batches[obs_idx].append(detail)
    score_batch.append(score)
    iter_time_batch.append(iter_time)
    c_smooth_batch.append(c_smooth.item())
    s_smooth_batch.append(s_smooth.item())

    # Count safe trials: safe only if both safe1 >= 0 AND safe2 >= 0
    print(safe1_val, safe2_val)
    if safe1_val >= 0 and safe2_val >= 0:
        num_safe_total += 1

    if safe1_val < 0 or safe2_val < 0:
        for obs_idx, detail in enumerate(planned_details):
            print(
                f"[SafetyDebug] iter={it:03d} obs={obs_idx+1} order={detail['order']} "
                f"center=({detail['center_x']:.2f},{detail['center_y']:.2f}) "
                f"min={detail['min_cbf']:.4f} at t={detail['min_t']} "
                f"pos(y,x)=({detail['min_pos_y']:.3f},{detail['min_pos_x']:.3f}) "
                f"violating_steps={detail['num_violating']}"
            )
    ##################################################end statistics calculation

    # logger.finish(t, env.max_episode_steps, score=score, value=0)

    ##################################################start per-iter statistics calculation
    # per-iter statistics calculation
    score_val    = float(score)
    itertime_val = float(iter_time[0])
    success_val  = 1 if reward > 0.95 else 0

    # for time saving skip it
    # append mode to write one line to csv file
    # with open(csv_path, 'a', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow([
    #         iter,
    #         safe1_val,
    #         safe2_val,
    #         is_trap1,
    #         is_trap2,
    #         score_val,
    #         itertime_val,
    #         success_val
    #     ])

    # Track planning time only
    plan_total_time_batch.append(plan_total_time)
    ##################################################end per-iter statistics calculation

# --------------------------------- statistics calculation ----------------------------------#

iter_time_batch = np.array(iter_time_batch)
iter_time_avg = np.mean(iter_time_batch, axis=0)

# Calculate average planning time
plan_total_time_batch = np.array(plan_total_time_batch)
avg_plan_time = np.mean(plan_total_time_batch)

# Calculate safe_total_rate
safe_total_rate = num_safe_total / TOTAL_TEST_ITER if TOTAL_TEST_ITER > 0 else 0.0

# pdb.set_trace()
print("======================results======================")
# elbo_batch = np.array(elbo_batch)
# print("elbo mean: ", np.mean(elbo_batch))
# print("elbo std: ", np.std(elbo_batch))

score_batch = np.array(score_batch)
print(f"safe1: min: {np.min(safe1_batch):.4f}/ {np.mean(safe1_batch):.4f} ± {np.std(safe1_batch):.4f}")
print(f"safe2: min: {np.min(safe2_batch):.4f}/ {np.mean(safe2_batch):.4f} ± {np.std(safe2_batch):.4f}")
if len(safe_detail_batches[0]) > 0:
    obs1_viol_steps = np.array([d['num_violating'] for d in safe_detail_batches[0]], dtype=float)
    obs2_viol_steps = np.array([d['num_violating'] for d in safe_detail_batches[1]], dtype=float)
    print(f"obs1 violating steps: mean {obs1_viol_steps.mean():.1f}, max {int(obs1_viol_steps.max())}")
    print(f"obs2 violating steps: mean {obs2_viol_steps.mean():.1f}, max {int(obs2_viol_steps.max())}")
print(f"safe_total_rate: {num_safe_total} / {TOTAL_TEST_ITER} = {safe_total_rate:.4f}")
print(f"trap1: {num_trap1} / {TOTAL_TEST_ITER}")
print(f"trap2: {num_trap2} / {TOTAL_TEST_ITER}")
print(f"c-smooth: {np.mean(c_smooth_batch):.4f} ± {np.std(c_smooth_batch):.4f}")
print(f"s-smooth: {np.mean(s_smooth_batch):.4f} ± {np.std(s_smooth_batch):.4f}")
print(f"score: {np.mean(score_batch):.5f} ± {np.std(score_batch):.5f}")
print(f"number of success: {num_success} / {TOTAL_TEST_ITER}")
if len(aegis_psi_batch) > 0:
    print(f"aegis-psi: {np.mean(aegis_psi_batch):.5f} ± {np.std(aegis_psi_batch):.5f}")
    print(f"aegis-barrier: {np.mean(aegis_barrier_batch):.5f} ± {np.std(aegis_barrier_batch):.5f}")
    print(f"aegis-delta: {np.mean(aegis_delta_batch):.5f} ± {np.std(aegis_delta_batch):.5f}")
print("--------------------- timing ---------------------")
print(f"[Plan] Total time (avg): {avg_plan_time*1000:.2f} ms ± {np.std(plan_total_time_batch)*1000:.2f} ms")
print("=======================end=========================")

# --------------------------------- plot ----------------------------------#
try:
    import matplotlib.pyplot as plt
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot safe1 data
    ax1.set_title('Safety Specification 1')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Safety Value')
    ax1.plot(range(len(safe1_batch)), safe1_batch, 'b-', label='Safe1')
    ax1.grid(True)
    ax1.legend()
    
    # Plot safe2 data
    ax2.set_title('Safety Specification 2')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Safety Value')
    ax2.plot(range(len(safe2_batch)), safe2_batch, 'r-', label='Safe2')
    ax2.grid(True)
    ax2.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    imgpath = join(args.savepath, 'safety_stats.png')
    plt.savefig(imgpath, dpi=300, bbox_inches='tight')
    plt.close('all')  # Properly close all figures
    
except Exception as e:
    print(f"Warning: Could not create plots: {str(e)}")

# Save result as a json file
# try:
json_path = join(args.savepath, 'rollout.json')
json_data = {
    'peak_memory': torch.cuda.max_memory_allocated() / (1024 ** 2),
    'safety_stats': {
        'safe1': safe1_batch,
        'safe2': safe2_batch,
        'safe1_mean': float(np.mean(safe1_batch)),
        'safe2_mean': float(np.mean(safe2_batch)),
        'safe1_std': float(np.std(safe1_batch)),
        'safe2_std': float(np.std(safe2_batch)),
        'safe1_min': float(np.min(safe1_batch)),
        'safe2_min': float(np.min(safe2_batch)),
        'safe_total_count': num_safe_total,
        'safe_total_rate': float(safe_total_rate)
    },
    'smooth_stats': {
        'c_smooth': c_smooth_batch,
        's_smooth': s_smooth_batch,
        'c_smooth_mean': float(np.mean(c_smooth_batch)),
        's_smooth_mean': float(np.mean(s_smooth_batch)),
        'c_smooth_std': float(np.std(c_smooth_batch)),
        's_smooth_std': float(np.std(s_smooth_batch)),
    },
    'score_stats': {
        'score': score_batch.tolist(),
        'mean': float(np.mean(score_batch)),
        'std': float(np.std(score_batch)),
        'min': float(np.min(score_batch)),
        'max': float(np.max(score_batch))
    },
    'planning_time_ms': {
        'time_batch': (plan_total_time_batch * 1000).tolist(),
        'mean': float(avg_plan_time * 1000),
        'std': float(np.std(plan_total_time_batch) * 1000),
        'min': float(np.min(plan_total_time_batch) * 1000),
        'max': float(np.max(plan_total_time_batch) * 1000)
    },
    'aegis_stats': {
        'enabled': bool(getattr(args, 'aegis_enabled', False)),
        'psi_mean': float(np.mean(aegis_psi_batch)) if len(aegis_psi_batch) > 0 else 0.0,
        'barrier_mean': float(np.mean(aegis_barrier_batch)) if len(aegis_barrier_batch) > 0 else 0.0,
        'control_delta_mean': float(np.mean(aegis_delta_batch)) if len(aegis_delta_batch) > 0 else 0.0,
    },
    'local_trap1': f'{num_trap1}/{TOTAL_TEST_ITER}',
    'local_trap2': f'{num_trap2}/{TOTAL_TEST_ITER}',
    'denoising_step': args.n_diffusion_steps,
    # 'step': int(t),
    # 'return': float(total_reward),
    # 'term': bool(terminal),
    # 'epoch_diffusion': int(diffusion_experiment.epoch),
}
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2, sort_keys=True)
# except Exception as e:
#     print(f"Warning: Could not save json file: {str(e)}")
