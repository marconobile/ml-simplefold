#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
from tqdm import tqdm
from einops import repeat
from utils.boltz_utils import center_random_augmentation


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
        conditioning_key="atom_idx_and_glob_cluster_id_per_frame",
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

    def create_unconditioned_batch(self, batch):
        if not isinstance(batch, dict) or self.conditioning_key not in batch:
            return None

        unconditioned_batch = dict(batch)
        unconditioned_batch[self.conditioning_key] = torch.full_like(
            batch[self.conditioning_key],
            -1,
        )
        return unconditioned_batch

    def should_use_guidance(self, batch_uncond=None, model_fn_conditioned=None, c=None):
        if self.guidance_scale is None or float(self.guidance_scale) == 1.0:
            return False
        return batch_uncond is not None or (
            model_fn_conditioned is not None and c is not None
        )

    def diffusion_coefficient(self, t, eps=0.01):
        # determine diffusion coefficient
        w = (1.0 - t) / (t + eps)
        if t >= self.w_cutoff:
            w = 0.0
        return w

    @torch.no_grad()
    def euler_maruyama_step(
        self,
        model_fn,
        flow,
        y, 
        t, 
        t_next, 
        batch, 
        batch_uncond=None,
        model_fn_conditioned=None,
        c=None,
    ):
        dt = t_next - t
        eps = torch.randn_like(y).to(y)

        y = center_random_augmentation(
            y,
            batch["atom_pad_mask"],
            augmentation=False,
            centering=True,
        )

        batched_t = repeat(t, " -> b", b=y.shape[0])
        use_guidance = self.should_use_guidance(
            batch_uncond=batch_uncond,
            model_fn_conditioned=model_fn_conditioned,
            c=c,
        )
        if use_guidance and batch_uncond is not None:
            velocity_uncond = model_fn(
                noised_pos=y,
                t=batched_t,
                feats=batch_uncond,
            )["predict_velocity"]
            velocity_cond = model_fn(
                noised_pos=y,
                t=batched_t,
                feats=batch,
            )["predict_velocity"]
            velocity = velocity_uncond + self.guidance_scale * (
                velocity_cond - velocity_uncond
            )
        elif use_guidance:
            velocity_uncond = model_fn(
                noised_pos=y,
                t=batched_t,
                feats=batch,
            )["predict_velocity"]
            if hasattr(c, "to"):
                c = c.to(y)
            velocity_cond = model_fn_conditioned(
                noised_pos=y,
                t=batched_t,
                feats=batch,
                c=c,
            )["predict_velocity"]
            velocity = velocity_uncond + self.guidance_scale * (
                velocity_cond - velocity_uncond
            )
        else:
            velocity = model_fn(
                noised_pos=y,
                t=batched_t,
                feats=batch,
            )["predict_velocity"]

        score = flow.compute_score_from_velocity(velocity, y, t)

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
        batch_uncond = self.create_unconditioned_batch(feats)
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
                batch_uncond=batch_uncond,
                model_fn_conditioned=model_fn_conditioned,
                c=c,
            )

        return {
            "denoised_coords": y_sampled
        }
