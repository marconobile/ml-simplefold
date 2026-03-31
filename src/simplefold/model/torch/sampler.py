#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

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
        use_exendiff = True
        if use_exendiff:
            # S_rest(x_t,t) = s_theta(x_t,t) - 0.5 * k * grad_{x_t} || y_target - A(x0_hat) ||_2^2
            # for this linear FM path: x0_hat = y_t + (1 - t) * v_theta(y_t, t)
            target_atom_coords = batch.get("ref_pos") #! introduce reference data + matching atom name
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
                residual_l2_norm = torch.sum(residual * residual)
                k = (2/t**2)/residual.abs()
                conditioning_loss = 0.5 * k * residual_l2_norm

                grad_conditioning = torch.autograd.grad(
                    conditioning_loss,
                    y_for_grad,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )[0]

            score = score - grad_conditioning
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
