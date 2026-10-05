import os
import time
import json
from absl import flags
import torch

import RectifiedFlow.datasets as datasets

import torch.nn as nn
import sys
from RectifiedFlow.utils import load_torch_file, save_checkpoint, restore_checkpoint
import numpy as np
from PIL import Image
import cv2
import torch.nn.functional as F

from .method_utils import (
    get_img,
    embed_to_latent,
    save_img,
    generate_traj,
    generate_traj_with_guidance,
)


import warnings

warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS

class AlignedArcFaceIdentityEncoder(nn.Module):
    """
    One-time landmark alignment + differentiable crop for rollout.
    - Detect landmarks once on source reference image (no grad needed).
    - Build fixed theta for grid_sample.
    - During forward, apply differentiable affine warp -> 112x112 -> ArcFace.
    """
    def __init__(self, insightface_repo, weight_path, device, source_image_tensor):
        super().__init__()
        self.device = device
        self.insightface_repo = insightface_repo

        if insightface_repo not in sys.path:
            sys.path.insert(0, insightface_repo)

        from model import Backbone
        from mtcnn import MTCNN
        from mtcnn_pytorch.src.align_trans import get_reference_facial_points

        self.backbone = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se').to(device)
        state = load_torch_file(weight_path, map_location=device, trusted=True)
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.out_h = 112
        self.out_w = 112

        # Build fixed alignment theta from source image once.
        mtcnn = self._init_mtcnn(MTCNN)
        self.theta = self._build_fixed_theta(
            source_image_tensor=source_image_tensor,
            mtcnn=mtcnn,
            get_reference_facial_points_fn=get_reference_facial_points,
        ).to(device)

    def _init_mtcnn(self, mtcnn_cls):
      prev_cwd = os.getcwd()
      np_load_old = np.load

      def np_load_allow_pickle(*args, **kwargs):
        kwargs.setdefault('allow_pickle', True)
        return np_load_old(*args, **kwargs)

      try:
        os.chdir(self.insightface_repo)
        np.load = np_load_allow_pickle
        return mtcnn_cls()
      except Exception as exc:
        print(f"[AlignedArcFaceIdentityEncoder] MTCNN init failed, fallback to identity crop: {exc}")
        return None
      finally:
        np.load = np_load_old
        os.chdir(prev_cwd)

    @torch.no_grad()
    def _build_fixed_theta(self, source_image_tensor, mtcnn, get_reference_facial_points_fn):
      """
      source_image_tensor: shape [B,3,H,W], range [0,1]
      Returns theta shape [B,2,3] for affine_grid (align_corners=False).
      """
      assert source_image_tensor.dim() == 4
      B, _, H, W = source_image_tensor.shape

      dst_pts = np.array(get_reference_facial_points_fn(default_square=True), dtype=np.float32)

      def n_to_p(width, height):
        return np.array([
          [width / 2.0, 0.0, (width - 1.0) / 2.0],
          [0.0, height / 2.0, (height - 1.0) / 2.0],
          [0.0, 0.0, 1.0]
        ], dtype=np.float32)

      def p_to_n(width, height):
        return np.linalg.inv(n_to_p(width, height))

      p_out = n_to_p(self.out_w, self.out_h)
      p_in_inv = p_to_n(W, H)

      thetas = []
      eye_theta = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

      for b in range(B):
        x = source_image_tensor[b].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        x = (x * 255.0).astype(np.uint8)
        pil = Image.fromarray(x)

        if mtcnn is None:
          landmarks = None
        else:
          try:
            _, landmarks = mtcnn.detect_faces(pil)
          except Exception:
            landmarks = None

        if landmarks is None or len(landmarks) == 0:
          thetas.append(eye_theta)
          continue

        lm = landmarks[0]
        src_pts = np.array([[lm[i], lm[i + 5]] for i in range(5)], dtype=np.float32)

        m, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
        if m is None:
          thetas.append(eye_theta)
          continue

        m3 = np.eye(3, dtype=np.float32)
        m3[:2, :] = m
        minv = np.linalg.inv(m3)

        t = p_in_inv @ minv @ p_out  # output_norm -> input_norm
        thetas.append(t[:2, :].astype(np.float32))

      theta = torch.from_numpy(np.stack(thetas, axis=0)).float()
      return theta

    def _align_crop(self, x):
        # x: [B,3,H,W], [0,1]
        B = x.shape[0]
        if self.theta.shape[0] == 1 and B > 1:
            theta = self.theta.expand(B, -1, -1).contiguous()
        elif self.theta.shape[0] != B:
            raise ValueError(f"theta batch {self.theta.shape[0]} does not match input batch {B}")
        else:
            theta = self.theta.contiguous()

        grid = F.affine_grid(theta, size=(B, 3, self.out_h, self.out_w), align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return x

    def forward(self, x):
        # x: [B,3,H,W], expected [0,1]
        x = x.clamp(0.0, 1.0)
        x = self._align_crop(x)
        # InsightFace test normalization
        x = (x - 0.5) / 0.5
        emb = self.backbone(x)         # Backbone already l2 normalizes
        emb = F.normalize(emb, dim=1)  # keep explicit normalization
        return emb

def sanitize_prompt_for_filename(prompt):
    prompt_mapping = {
        "A photo of an old face.": "old",
        "A photo of a sad face.": "sad",
        "A photo of a smiling face.": "smile",
        "A photo of an angry face.": "angry",
        "A photo of a face with curly hair.": "curly",
    }
    return prompt_mapping.get(
        prompt,
        prompt.replace("A photo of ", "").replace(".", "").replace(" ", "_").lower(),
    )


def oc_flow_optimization(
    z0,
    u_ind,
    dynamics,
    generate_traj,
    L_N,
    x_ref,  # reference image for LPIPS
    lpips_model,
    N=100,
    number_of_iterations=25,
    lr=2.5,
    lpips_threshold=0.1,
    constraint_margin=0.0,
    weight_decay=0.995,
):
    device, shape = z0.device, z0.shape
    batch_size = shape[0]
    assert batch_size == 1

    eps = 1e-3
    dt = 1.0 / N

    # helper: penalty for constraint violation
    def penalty(xN):

        # constraint: g(xN) = LPIPS(xN, x_ref) - threshold + margin <= 0
        d = lpips_model(xN, x_ref).view(xN.shape[0])
        g = d - lpips_threshold + constraint_margin

        hinge = torch.clamp(g, min=0.0)

        return (hinge**2).mean()

    u = {j: torch.zeros_like(z0, device=device) for j in u_ind}
    for j in u:
        u[j].requires_grad = False
        u[j].grad = torch.zeros_like(u[j], device=device)

    L_best, opt_u = -float("inf"), {j: u[j].clone() for j in u}

    for iter in range(number_of_iterations):
        z_traj = generate_traj(dynamics, z0, u=u, N=N)

        xN = z_traj[-1].to(device).detach().clone().requires_grad_(True)

        V_N = -L_N(xN) - penalty(xN)
        lam = torch.autograd.grad(V_N, xN)[0].detach().clone()  # \lambda_N

        L_curr = V_N.item()
        if L_curr > L_best:
            L_best = L_curr
            for j in u:
                opt_u[j] = u[j].detach().clone()

        for j in reversed(range(N)):  # N-1, ..., 0

            lam_next = lam  # \lambda_{j+1}

            xj = z_traj[j].to(device).detach().clone().requires_grad_(True)
            tj = torch.full((batch_size,), j / N * (1.0 - eps) + eps, device=device)

            def F_j(x_flat):
                x = x_flat.view(shape)
                return (x + dynamics(x, tj * 999) * dt).view(-1)

            _, vjp = torch.autograd.functional.vjp(
                F_j, xj.view(-1), v=lam_next.view(-1)
            )
            lam = vjp.view(shape).detach()  # \lambda_j

            u[j].grad = lam_next.detach().clone()

        for j in u:
            u[j] = weight_decay * u[j] + batch_size * lr * u[j].grad

    return opt_u

def _epsilon_t(t):
    d_start = 2.5
    d_end = .4
    t0 = .55
    beta = 10.

    # Logistic schedule on squared distance, then map to epsilon radius.
    q = 1.0 / (1.0 + torch.exp(beta * (t - t0)))
    d_t = d_end + (d_start - d_end) * q
    return float(torch.sqrt(torch.tensor(max(d_t, 1e-8))))

    # eps_min = cfg.get('epsilon_min', 0.22)
    # eps_max = cfg.get('epsilon_max', 0.40)
    # power = cfg.get('epsilon_power', 1.0)
    # return eps_min + ((1.0 - t) ** power) * (eps_max - eps_min)

def _barrier(h):
    h = torch.nan_to_num(h, nan=0.0, posinf=1e3, neginf=-1e3)
    barrier_gain = 1.
    barrier_alpha = .15
    barrier_delta = 0.
    barrier_slope = 5.
    denom_floor = 1e-2
    barrier_value_clip = 50.
  
    denom = torch.clamp(h + barrier_alpha, min=denom_floor)
    sigma = torch.sigmoid(-barrier_slope * (h - barrier_delta))
    barrier_val = barrier_gain * sigma / denom
    barrier_val = torch.nan_to_num(barrier_val, nan=barrier_value_clip, posinf=barrier_value_clip, neginf=0.0)
    barrier_val = torch.clamp(barrier_val, min=0.0, max=barrier_value_clip)
    return barrier_val


def aegisflow_optimization(
    z0,
    u_ind,
    dynamics,
    generate_traj,
    L_N,
    x_ref,  # reference image for LPIPS
    lpips_model,
    N=100,
    number_of_iterations=25,
    lr=2.5,
    lpips_threshold=0.1,
    constraint_margin=0.0,
    weight_decay=0.995,
):
    device, shape = z0.device, z0.shape
    batch_size = shape[0]
    assert batch_size == 1
    
    '''
    Aegis Parameters:
    '''
    gradBj_clip = 1.
    lamda = .6
    beta = .1
    lpips_threshold = 0.1

    arc_id = AlignedArcFaceIdentityEncoder(
        insightface_repo='/root/autodl-tmp/AegisFlow/third_party/InsightFace',
        weight_path='/root/autodl-tmp/AegisFlow/wt/arcface.pth',
        device=device,
        source_image_tensor=x_ref
    ).to(device)
    with torch.no_grad():
        src_emb = arc_id(x_ref)

    eps = 1e-3
    dt = 1.0 / N

    # helper: penalty for constraint violation
    def penalty(xN):

        # constraint: g(xN) = LPIPS(xN, x_ref) - threshold + margin <= 0
        d = lpips_model(xN, x_ref).view(xN.shape[0])
        g = d - lpips_threshold + constraint_margin

        hinge = torch.clamp(g, min=0.0)

        return 1.*(hinge**2).mean()

    u = {j: torch.zeros_like(z0, device=device) for j in u_ind}
    for j in u:
        u[j].requires_grad = False
        u[j].grad = torch.zeros_like(u[j], device=device)

    L_best, opt_u = -float("inf"), {j: u[j].clone() for j in u}

    for iter in range(number_of_iterations):
        z_traj = generate_traj(dynamics, z0, u=u, N=N)

        xN = z_traj[-1].to(device).detach().clone().requires_grad_(True)

        V_N = -L_N(xN) - penalty(xN)
        lam = torch.autograd.grad(V_N, xN)[0].detach().clone()  # \lambda_N

        L_curr = V_N.item()
        e = arc_id(xN) - src_emb
        d = (e * e).sum(dim=1).mean().item()
        # raise Exception('here')
        print(f"Iteration {iter+1}/{number_of_iterations}, L_N: {L_curr:.4f}, best: {L_best:.4f}, similarity: {1-d/2:.4f}")
        if L_curr > L_best:
            L_best = L_curr
            for j in u:
                opt_u[j] = u[j].detach().clone()

        for j in reversed(range(N)):  # N-1, ..., 0

            lam_next = lam  # \lambda_{j+1}

            xj = z_traj[j].to(device).detach().clone().requires_grad_(True)
            tj = torch.full((batch_size,), j / N * (1.0 - eps) + eps, device=device)
            eps_t = _epsilon_t(tj)

            emb = arc_id(xj)
            e = emb - src_emb
            d = (e * e).sum(dim=1)
            hj = (eps_t**2) - d
            Bj = _barrier(hj)
            grad_Bj = torch.autograd.grad(Bj.mean(), xj, create_graph=False, retain_graph=False)[0].detach()
            gradBj_flat = grad_Bj.view(batch_size, -1)
            gradBj_norm = torch.norm(gradBj_flat, dim=1, keepdim=True).clamp_min(1e-8)
            gradBj_scale = torch.clamp(gradBj_clip / gradBj_norm, max=1.0)
            grad_Bj = (gradBj_flat * gradBj_scale).view_as(grad_Bj)

            sj = (1-tj)/lamda

            def F_j(x_flat):
                x = x_flat.view(shape)
                return (x + dynamics(x, tj * 999) * dt).view(-1)

            _, vjp = torch.autograd.functional.vjp(
                F_j, xj.view(-1), v=lam_next.view(-1)
            )
            lam = vjp.view(shape).detach()  # \lambda_j

            # u[j].grad = lam_next.detach().clone()
            u[j].grad = -sj*beta*grad_Bj.detach().clone() + lam_next.detach().clone()/lamda

        for j in u:
            u[j] = weight_decay * u[j] + batch_size * lr * u[j].grad

    return opt_u

def hardflow_optimization(
    z0,
    dynamics,
    L_N,
    x_ref,  # reference image for LPIPS
    lpips_model,
    N=100,
    gradient_steps=25,
    dual_update_frequency=5,
    lr=2.5,
    lpips_threshold=0.1,
    constraint_margin=0.0,
    rho=5.0,
    rho_mult=1.5,
    regularization=0.0,
):
    device, shape = z0.device, z0.shape
    batch_size = shape[0]
    assert batch_size == 1

    eps = 1e-3
    dt = 1.0 / N

    z = z0.detach().clone()
    traj = []
    traj.append(z.detach().clone())

    arc_id = AlignedArcFaceIdentityEncoder(
        insightface_repo='/root/autodl-tmp/AegisFlow/third_party/InsightFace',
        weight_path='/root/autodl-tmp/AegisFlow/wt/arcface.pth',
        device=device,
        source_image_tensor=x_ref
    ).to(device)
    dist_threshold = 0.35
    src_emb = arc_id(x_ref)

    # helper: terminal augmented Lagrangian value
    def V_AL(xN, lam_dual_given, rho_given):
        V_base = -L_N(xN)

        # constraint: g(xN) = LPIPS(xN, x_ref) - threshold + margin <= 0
        # d = lpips_model(xN, x_ref).view(xN.shape[0])
        # g = d - lpips_threshold + constraint_margin
        e = arc_id(xN) - src_emb
        d = (e * e).sum(dim=1)
        g = d - dist_threshold

        hinge = torch.clamp(g, min=0.0)

        return V_base - lam_dual_given * g.mean() - 0.5 * rho_given * (hinge**2).mean()

    for i in range(N):
        t = torch.ones(batch_size, device=device) * i / N * (1.0 - eps) + eps

        with torch.no_grad():
            vt = dynamics(z, t * 999)
            z_prediction_ref = z + vt * (1.0 - t)
            z_next_ref = z + vt * dt

        inputs = z_prediction_ref.detach().requires_grad_(True)

        use_constraint = i >= N // 2
        lam_dual = torch.tensor(0.0, device=device) if use_constraint else None
        rho_step = rho if use_constraint else None

        def get_objective(inputs_val, lam=None, rho_val=None):
            reg_term = (
                (t + dt) ** 2
                * ((inputs_val - z_prediction_ref) ** 2).mean()
                * regularization
            )
            if use_constraint:
                return V_AL(inputs_val, lam, rho_val) + reg_term
            else:
                return -L_N(inputs_val) + reg_term

        for j in range(gradient_steps):
            objective = get_objective(inputs, lam_dual, rho_step)
            (grad,) = torch.autograd.grad(
                objective,
                inputs,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )
            inputs = (inputs + lr * grad).detach().requires_grad_(True)
            del grad

            if use_constraint and (j + 1) % dual_update_frequency == 0:
                with torch.no_grad():
                    # d_cur = lpips_model(inputs, x_ref).view(inputs.shape[0])
                    # g_cur = float(
                    #     (d_cur - lpips_threshold + constraint_margin).mean().item()
                    # )
                    e_cur = arc_id(inputs) - src_emb
                    d_cur = (e_cur * e_cur).sum(dim=1)
                    g_cur = float((d_cur - dist_threshold).mean().item())
                lam_dual = torch.clamp(lam_dual + rho_step * g_cur, min=0.0)
                if g_cur > 0.02:
                    rho_step = rho_step * max(1.0, rho_mult)

        z = z_next_ref + (t + dt) * (inputs - z_prediction_ref)

        traj.append(z.detach().clone())

    e = arc_id(traj[-1]) - src_emb
    d = (e * e).sum(dim=1).mean().item()

    return traj, d


def gradient_guidance_optimization(
    z0,
    dynamics,
    L_N,
    x_ref,  # reference image for LPIPS
    lpips_model,
    N=100,
    gradient_steps=25,
    lr=2.5,
    lpips_threshold=0.1,
    constraint_margin=0.0,
):
    device, shape = z0.device, z0.shape
    batch_size = shape[0]
    assert batch_size == 1

    def penalty(xN):

        d = lpips_model(xN, x_ref).view(xN.shape[0])
        g = d - lpips_threshold + constraint_margin

        hinge = torch.clamp(g, min=0.0)

        return (hinge**2).mean()

    full_L_N = lambda x: L_N(x) + penalty(x)

    traj = generate_traj_with_guidance(
        dynamics, z0, N=N, L_N=full_L_N, alpha_L=lr, guidance_steps=gradient_steps
    )

    return traj


def projection_relaxed_optimization(
    z0,
    dynamics,
    L_N,
    x_ref,  # reference image for LPIPS
    lpips_model,
    N=100,
    gradient_steps=25,
    lr=2.5,
    lpips_threshold=0.1,
    constraint_margin=0.0,
    lr_al=2.5,
    rho=5.0,
    rho_mult=1.0,
    al_steps=5,
):
    device, shape = z0.device, z0.shape
    batch_size = shape[0]
    assert batch_size == 1

    def V_AL(x_new, x_current, lam_dual_given, rho_given):
        V_base = -((x_new - x_current) ** 2).mean()

        d = lpips_model(x_new, x_ref).view(x_new.shape[0])
        g = d - lpips_threshold + constraint_margin

        hinge = torch.clamp(g, min=0.0)

        return V_base - lam_dual_given * g.mean() - 0.5 * rho_given * (hinge**2).mean()

    eps = 1e-3
    dt = 1.0 / N

    z = z0.detach().clone()
    traj = [z.clone()]

    for i in range(N):
        t = (i / N) * (1.0 - eps) + eps
        t_vec = torch.full((batch_size,), t, device=device)
        t_b = t_vec.view(batch_size, *([1] * (z.ndim - 1)))
        use_constraint = i >= N // 2
        if i == N // 2:
            lam_dual = torch.tensor(0.0, device=device)

        if L_N is not None and gradient_steps > 0:
            for _ in range(gradient_steps):
                inputs = z.detach().clone().requires_grad_(True)
                vt = dynamics(inputs, t_vec * 999.0)
                loss = L_N(inputs + vt * (1.0 - t_b))
                if loss.ndim > 0:
                    loss = loss.mean()
                (g,) = torch.autograd.grad(loss, inputs, create_graph=False)

                z = z - lr * g.detach()

        if use_constraint:
            x_current = z.detach().clone()
            inputs = z.detach().clone().requires_grad_(True)
            for _ in range(al_steps):
                objective = V_AL(inputs, x_current, lam_dual, rho)
                (grad,) = torch.autograd.grad(
                    objective,
                    inputs,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )
                inputs = (inputs + lr_al * grad).detach().requires_grad_(True)
                del grad

            with torch.no_grad():
                d_cur = lpips_model(inputs, x_ref).view(inputs.shape[0])
                g_cur = float(
                    (d_cur - lpips_threshold + constraint_margin).mean().item()
                )

            lam_dual = torch.clamp(lam_dual + rho * g_cur, min=0.0)
            if g_cur > 0.02:
                rho = rho * max(1.0, rho_mult)

            z = inputs.detach().clone()

        with torch.no_grad():
            vt = dynamics(z, t_vec * 999.0).detach()
            z = z + vt * dt

        traj.append(z.clone())

    return traj


def image_edit(config, models, image_path, text_prompt, output_dir):

    # models
    model_fn = models["model_fn"]
    clip_loss_1 = models["clip_loss_1"]
    lpips_model = models["lpips_model"]

    N = 100

    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    prompt_suffix = sanitize_prompt_for_filename(text_prompt)
    opt_img_path = os.path.join(output_dir, f"{img_name}_{prompt_suffix}.png")
    original_img_path = os.path.join(output_dir, f"{img_name}.png")
    image = get_img(image_path)
    original_img = image.to(config.device)

    arc_id = AlignedArcFaceIdentityEncoder(
        insightface_repo='/root/autodl-tmp/AegisFlow/third_party/InsightFace',
        weight_path='/root/autodl-tmp/AegisFlow/wt/arcface.pth',
        device=FLAGS.config.device,
        source_image_tensor=original_img,
    ).to(FLAGS.config.device)

    clip_loss_1.set_ref_img(original_img)
    clip_loss_1.set_text(text_prompt)

    scaled_img = scaler(original_img)
    save_img(inverse_scaler(scaled_img), path=original_img_path)

    latent = embed_to_latent(model_fn, scaler(original_img))
    traj = generate_traj(model_fn, latent, N=N)

    print(f"Optimization starts: {image_path} -> {opt_img_path}")

    t_start = time.time()

    if FLAGS.method == "oc_flow":
        u_ind = [_ for _ in range(N)]
        u_opt = oc_flow_optimization(
            z0=latent,
            u_ind=u_ind,
            dynamics=model_fn,
            generate_traj=generate_traj,
            L_N=clip_loss_1.L_N,
            x_ref=traj[-1].detach(),
            lpips_model=lpips_model,
            N=N,
            number_of_iterations=25,
            lr=2.5,
            lpips_threshold=0.1,
            constraint_margin=0.02,
        )
        device = latent.device
        arc_id = AlignedArcFaceIdentityEncoder(
            insightface_repo='/root/autodl-tmp/AegisFlow/third_party/InsightFace',
            weight_path='/root/autodl-tmp/AegisFlow/wt/arcface.pth',
            device=device,
            source_image_tensor=traj[-1].detach()
        ).to(device)
        traj_opt = generate_traj(model_fn, z0=latent, u=u_opt, N=N)
        e = arc_id(traj_opt[-1]) - arc_id(traj[-1])
        d = (e * e).sum(dim=1).mean().item()
    elif "aegis" in FLAGS.method:
        u_ind = [_ for _ in range(N)]
        u_opt = aegisflow_optimization(
            z0=latent,
            u_ind=u_ind,
            dynamics=model_fn,
            generate_traj=generate_traj,
            L_N=clip_loss_1.L_N,
            x_ref=traj[-1].detach(),
            lpips_model=lpips_model,
            N=N,
            number_of_iterations=10,
            lr=5.,
            lpips_threshold=0.06,
            constraint_margin=0.02,
        )
        device = latent.device
        arc_id = AlignedArcFaceIdentityEncoder(
            insightface_repo='/root/autodl-tmp/AegisFlow/third_party/InsightFace',
            weight_path='/root/autodl-tmp/AegisFlow/wt/arcface.pth',
            device=device,
            source_image_tensor=traj[-1].detach()
        ).to(device)
        traj_opt = generate_traj(model_fn, z0=latent, u=u_opt, N=N)
        e = arc_id(traj_opt[-1]) - arc_id(traj[-1])
        d = (e * e).sum(dim=1).mean().item()
    elif FLAGS.method == "gradient_guidance":
        traj_opt = gradient_guidance_optimization(
            z0=latent,
            dynamics=model_fn,
            L_N=clip_loss_1.L_N,
            x_ref=traj[-1].detach(),
            lpips_model=lpips_model,
            N=N,
            gradient_steps=25,
            lr=2.5,
            lpips_threshold=0.06,
            constraint_margin=0.02,
        )
    elif FLAGS.method == "projection_relaxed":
        traj_opt = projection_relaxed_optimization(
            z0=latent,
            dynamics=model_fn,
            L_N=clip_loss_1.L_N,
            x_ref=traj[-1].detach(),
            lpips_model=lpips_model,
            N=N,
            gradient_steps=25,
            lr=2.5,
            lpips_threshold=0.06,
            constraint_margin=0.02,
            lr_al=2.5,
            rho=5.0,
            rho_mult=1.02,
            al_steps=10,
        )
    elif "hardflow" in FLAGS.method:
        print(f"Using hardflow optimization with method: {FLAGS.method}")
        traj_opt, d = hardflow_optimization(
            z0=latent,
            dynamics=model_fn,
            L_N=clip_loss_1.L_N,
            x_ref=traj[-1].detach(),
            lpips_model=lpips_model,
            N=N,
            gradient_steps=40,
            dual_update_frequency=8,
            lr=2.5,
            lpips_threshold=0.2,
            constraint_margin=0.02,
            rho=5.0,
            rho_mult=2.0,
            regularization=0.0005,
        )
        print(f'ArcFace distance squared: {d:.4f}')
    else:
        raise ValueError(f"Unknown optimization method: {FLAGS.method}")
        traj_opt = traj

    t_end = time.time()
    total_time = t_end - t_start

    print(f"Optimization ends")

    save_img(inverse_scaler(traj_opt[-1]), path=opt_img_path)

    with torch.no_grad():

        clip_loss_orig = clip_loss_1.L_N(traj[-1]).item()
        lpips_score_orig = lpips_model(traj[-1], traj[-1]).item()
        clip_loss_opt = clip_loss_1.L_N(traj_opt[-1]).item()
        lpips_score_opt = lpips_model(traj_opt[-1], traj[-1]).item()
        print(
            f"clip score (original): {-clip_loss_orig:.4f}, lpips score (original): {lpips_score_orig:.4f}\n"
            f"clip score (optimized): {-clip_loss_opt:.4f}, lpips score (optimized): {lpips_score_opt:.4f}\n"
            f"ArcFace distance squared: {d:.4f}\n"
            f"total time: {total_time:.4f} s"
        )

        metric = {
            "clip_score": float(-clip_loss_opt),
            "lpips_score": float(lpips_score_opt),
            "clip_score_original": float(-clip_loss_orig),
            "lpips_score_original": float(lpips_score_orig),
            "ArcFace_distance_squared": float(d),
            "ArcFace_similarity": float(1.0 - d/2),
            "time": float(total_time),
        }

    return metric
