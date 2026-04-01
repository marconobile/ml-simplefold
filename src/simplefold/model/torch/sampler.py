#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

from pathlib import Path
import torch
from tqdm import tqdm
from einops import repeat
from utils.boltz_utils import center_random_augmentation


def A_fn(x):
    return x

class EMSampler():
    """
    A Euler-Maruyama solver for SDEs.
    """
    def __init__(
        self,
        num_timesteps=500,
        t_start=1e-4,
        tau=0.3,
        log_timesteps=False,
        w_cutoff=0.99,
        guidance_scale=1.0,
        conditioning_key="c",
    ):
        self.num_timesteps = num_timesteps
        self.log_timesteps = log_timesteps
        self.t_start = t_start
        self.tau = tau
        self.w_cutoff = w_cutoff
        self.guidance_scale = guidance_scale
        self.conditioning_key = conditioning_key

        if self.log_timesteps:
            t = 1.0 - torch.logspace(-2, 0, self.num_timesteps + 1).flip(0)
            t = t - torch.min(t)
            t = t / torch.max(t)
            self.steps = t.clamp(min=self.t_start, max=1.0)
        else:
            self.steps = torch.linspace(
                self.t_start, 1.0, steps=self.num_timesteps + 1
            )

    def _log_exendiff(self, t, score_og, score_new, grad_conditioning_og, grad_conditioning_new, scale):

        log_path = Path("/home/nobilm@usi.ch/ml-simplefold/log_score.txt")

        score_og_norm = torch.norm(score_og, p=2).item()
        score_new_norm = torch.norm(score_new, p=2).item()
        grad_conditioning_og_norm = torch.norm(grad_conditioning_og, p=2).item()
        grad_conditioning_new_norm = torch.norm(grad_conditioning_new, p=2).item()

        with log_path.open("a") as f:
            f.write(
                f"t: {t:.6f}, "
                f"score_og norm: {score_og_norm:.6f}, "
                f"score_new norm: {score_new_norm:.6f}, "
                f"grad_conditioning_og norm: {grad_conditioning_og_norm:.6f}, "
                f"grad_conditioning_new norm: {grad_conditioning_new_norm:.6f}, "
                f"scale: {scale:.8f}\n"
            )

    def diffusion_coefficient(self, t, eps=0.01):
        # determine diffusion coefficient
        w = (1.0 - t) / (t + eps)
        if t >= self.w_cutoff:
            w = 0.0
        return w


    def euler_maruyama_step(
        self,
        model_fn,
        flow,
        y,
        t,
        t_next,
        batch,
    ):
        dt = t_next - t
        eps = torch.randn_like(y).to(y)

        atom_pad_mask = batch["atom_pad_mask"]
        atom_pad_mask_3d = atom_pad_mask[..., None].to(y)

        # work in the same centered frame used by the model
        y = center_random_augmentation(
            y,
            atom_pad_mask,
            augmentation=False,
            centering=True,
        )

        batched_t = repeat(t, " -> b", b=y.shape[0]).to(y)

        # base model prediction and score
        with torch.no_grad():
            velocity = model_fn(
                noised_pos=y,
                t=batched_t,
                feats=batch,
            )["predict_velocity"]

            score = flow.compute_score_from_velocity(velocity, y, t)

        #* Apply exendiff conditioning:
        target_atom_coords = batch.get("target_atom_coords_aligned")
        use_exendiff = True #True and (target_atom_coords is not None)
        if use_exendiff:
            # S_rest(x_t,t) = s_theta(x_t,t) - 0.5 * k * grad_{x_t} || y_target - A(x0_hat) ||_2^2
            # for this linear FM path: x0_hat = y_t + (1 - t) * v_theta(y_t, t)
            with torch.enable_grad():
                y_for_grad = y.detach().requires_grad_(True)

                batched_t_grad = repeat(t, " -> b", b=y_for_grad.shape[0]).to(y_for_grad)

                velocity_grad = model_fn(
                    noised_pos=y_for_grad,
                    t=batched_t_grad,
                    feats=batch,
                )["predict_velocity"]

                t_pad = flow.right_pad_dims_to(y_for_grad, batched_t_grad)
                x0_hat = y_for_grad + (1.0 - t_pad) * velocity_grad

                # center target into the same frame as the model input / x0_hat
                target_atom_coords = target_atom_coords.to(y_for_grad)
                target_centered = center_random_augmentation(
                    target_atom_coords,
                    atom_pad_mask,
                    augmentation=False,
                    centering=True,
                )

                residual = (target_centered - A_fn(x0_hat)) * atom_pad_mask_3d
                l1 = torch.norm(score, p=1)
                k = 5/l1
                residual_l2_norm = torch.norm(residual, p=2)
                conditioning_loss = 0.5 * residual_l2_norm

                grad_conditioning = torch.autograd.grad(
                    conditioning_loss,
                    y_for_grad,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )[0]

            grad_conditioning_og = grad_conditioning.clone()
            score_og = score.clone()

            scale = score.norm() / (grad_conditioning.norm() + 1e-8)

            grad_conditioning = grad_conditioning * scale
            score = score - 2 * grad_conditioning
            self._log_exendiff(t, score_og, score, grad_conditioning_og, grad_conditioning, scale)

            # t, score_og, score_new, grad_conditioning_og, grad_conditioning_new, scale):


        #* End of exendiff conditioning.

        diff_coeff = self.diffusion_coefficient(t)
        drift = velocity + diff_coeff * score
        mean_y = y + drift * dt
        y_sample = mean_y + torch.sqrt(2.0 * dt * diff_coeff * self.tau) * eps

        return y_sample

    @torch.no_grad()
    def sample(
        self,
        model_fn,
        flow,
        noise,
        batch,
        model_fn_conditioned=None,
        c=None,
    ):
        sampling_timesteps = self.num_timesteps
        steps = self.steps.to(noise.device)
        y_sampled = noise
        feats = batch
        if c is None and isinstance(feats, dict):
            c = feats.get(self.conditioning_key, None)

        for i in tqdm(
            range(sampling_timesteps),
            desc="Sampling",
            total=sampling_timesteps,
        ):
            t = steps[i]
            t_next = steps[i + 1]

            y_sampled = self.euler_maruyama_step(
                model_fn,
                flow,
                y_sampled,
                t,
                t_next,
                feats,
            )
        return {
            "denoised_coords": y_sampled
        }
