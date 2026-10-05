import os
import imageio
import clip
import torch
import torchvision
import numpy as np

from RectifiedFlow.models.utils import from_flattened_numpy, to_flattened_numpy
from scipy import integrate

from .DiffAugment_pytorch import DiffAugment


@torch.no_grad()
def embed_to_latent(model_fn, img):
    device = img.device

    def ode_func(t, x):
        x = from_flattened_numpy(x, img.shape).to(device).type(torch.float32)
        vec_t = torch.ones(img.shape[0], device=x.device) * t
        drift = model_fn(x, vec_t * 999)
        return to_flattened_numpy(drift)

    rtol = atol = 1e-5
    method = "RK45"
    eps = 1e-3

    x = img.detach().clone()

    solution = integrate.solve_ivp(
        ode_func, (1.0, eps), to_flattened_numpy(x), rtol=rtol, atol=atol, method=method
    )
    nfe = solution.nfev
    x = (
        torch.tensor(solution.y[:, -1])
        .reshape(img.shape)
        .to(device)
        .type(torch.float32)
    )

    return x


@torch.no_grad()
def generate_traj(dynamics, z0, u=None, N=100):
    device, shape = z0.device, z0.shape
    batch_size = shape[0]
    assert batch_size == 1

    eps = 1e-3
    dt = 1.0 / N

    z = z0.detach().clone()
    traj = []
    traj.append(z.detach().clone())

    for i in range(N):
        t = torch.ones(z0.shape[0], device=z0.device) * i / N * (1.0 - eps) + eps
        vt = dynamics(z, t * 999)
        z = z.detach().clone() + vt * dt

        if u is not None:
            z = z + u[i]

        traj.append(z.detach().clone())

    return traj


def generate_traj_with_guidance(
    dynamics, z0, N=100, L_N=None, alpha_L=1.0, guidance_steps=1
):
    device, shape = z0.device, z0.shape
    batch_size = shape[0]
    assert batch_size == 1

    eps = 1e-3
    dt = 1.0 / N

    z = z0.detach().clone()
    traj = [z.clone()]

    for i in range(N):
        t = (i / N) * (1.0 - eps) + eps
        t_vec = torch.full((batch_size,), t, device=device)
        t_b = t_vec.view(batch_size, *([1] * (z.ndim - 1)))

        if L_N is not None and guidance_steps > 0:
            for _ in range(guidance_steps):
                inputs = z.detach().clone().requires_grad_(True)
                vt = dynamics(inputs, t_vec * 999.0)
                loss = L_N(inputs + vt * (1.0 - t_b))
                if loss.ndim > 0:
                    loss = loss.mean()
                (g,) = torch.autograd.grad(loss, inputs, create_graph=False)

                z = z - alpha_L * g.detach()

        with torch.no_grad():
            vt = dynamics(z, t_vec * 999.0).detach()
            z = z + vt * dt

        traj.append(z.clone())

    return traj


def get_img(path=None, device=None):
    img = imageio.imread(path)
    img = img / 255.0
    img = img[np.newaxis, :, :, :]
    img = img.transpose(0, 3, 1, 2)

    print(f"\nRead image from: {path}, image range: [{img.min():.1f}, {img.max():.1f}]")

    img = torch.tensor(img).float()
    img = torch.nn.functional.interpolate(img, size=256)

    return img


def save_img(img, path=None):
    torchvision.utils.save_image(
        img.clamp_(0.0, 1.0), os.path.join(path), nrow=16, normalize=False
    )


class clip_semantic_loss:
    def __init__(self, device, alpha=0.5, replicate=20, inverse_scaler=None):

        self.device = device
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(self.device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(self.device)
        clip_mode = "ViT-B/32"
        self.interp_mode = "bilinear"
        self.clip_model, _ = clip.load(clip_mode, device=self.device)
        self.clip_c = self.clip_model.logit_scale.exp()
        self.policy = "color,translation,resize,cutout"
        self.replicate = replicate
        self.alpha = alpha
        self.inverse_scaler = inverse_scaler

        self.text_tok = None
        self.img = None

    def set_ref_img(self, img):
        self.img = img

    def set_text(self, text):
        self.text_tok = clip.tokenize([text]).to(self.device)

    def L_N(self, x):
        sim = (self.inverse_scaler(x) - self.img).abs().mean()
        batch_size = self.img.shape[0]

        img_aug = DiffAugment(x.repeat(self.replicate, 1, 1, 1), policy=self.policy)
        img_aug = self.inverse_scaler(img_aug)
        img_aug = torch.nn.functional.interpolate(
            img_aug, size=224, mode=self.interp_mode
        )
        img_aug.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])

        logits_per_image, logits_per_text = self.clip_model(img_aug, self.text_tok)
        logits_per_image = logits_per_image.view(batch_size, self.replicate, -1).mean(
            dim=1
        )
        logits_per_image = logits_per_image / self.clip_c
        concept_loss = (-1.0) * logits_per_image

        return self.alpha * concept_loss.mean() + (1.0 - self.alpha) * sim.sum()
