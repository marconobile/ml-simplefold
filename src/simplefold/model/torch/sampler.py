#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

from pathlib import Path
import torch
from tqdm import tqdm
from einops import repeat
from utils.boltz_utils import center_random_augmentation



def A_fn(x, dihedral_atom_indices, dihedral_mask, eps=1e-8):
    """
    Compute masked dihedral angles from coordinates.

    Args:
        x: [N, 3] or [B, N, 3] coordinates.
        dihedral_atom_indices: [R, D, 4] or [B, R, D, 4] atom indices.
        dihedral_mask: [R, D] or [B, R, D] bool/0-1 mask.

    Returns:
        Dihedral angles in radians with masked-out entries set to 0.
        Shape: [R, D] if input x is [N, 3], else [B, R, D].
    """
    squeeze_batch = x.ndim == 2
    if squeeze_batch:
        x = x.unsqueeze(0)

    B, N, _ = x.shape

    dihedral_atom_indices = torch.as_tensor(
        dihedral_atom_indices,
        device=x.device,
        dtype=torch.long,
    )
    dihedral_mask = torch.as_tensor(
        dihedral_mask,
        device=x.device,
        dtype=torch.bool,
    )

    if dihedral_atom_indices.ndim == 3:
        dihedral_atom_indices = dihedral_atom_indices.unsqueeze(0).expand(B, -1, -1, -1)
    if dihedral_mask.ndim == 2:
        dihedral_mask = dihedral_mask.unsqueeze(0).expand(B, -1, -1)

    if dihedral_atom_indices.shape[-1] != 4 or dihedral_atom_indices.shape[:-1] != dihedral_mask.shape:
        raise ValueError(
            "dihedral_atom_indices and dihedral_mask must align on batch/residue/dihedral dimensions."
        )

    active_indices = dihedral_atom_indices[dihedral_mask]
    if active_indices.numel() > 0:
        if (active_indices < 0).any() or (active_indices >= N).any():
            raise IndexError("Active dihedral atom index out of bounds.")

    # Keep masked entries valid for gather; they are zeroed out in the output anyway.
    safe_indices = torch.where(
        dihedral_mask[..., None],
        dihedral_atom_indices,
        torch.zeros_like(dihedral_atom_indices),
    )

    batch_idx = torch.arange(B, device=x.device)[:, None, None, None]
    atoms = x[batch_idx, safe_indices]  # [B, R, D, 4, 3]
    p0, p1, p2, p3 = atoms.unbind(dim=-2)

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1_hat = b1 / torch.linalg.norm(b1, dim=-1, keepdim=True).clamp_min(eps)
    v = b0 - (b0 * b1_hat).sum(dim=-1, keepdim=True) * b1_hat
    w = b2 - (b2 * b1_hat).sum(dim=-1, keepdim=True) * b1_hat

    x_comp = (v * w).sum(dim=-1)
    y_comp = (torch.cross(b1_hat, v, dim=-1) * w).sum(dim=-1)
    angles = torch.atan2(y_comp, x_comp)  # [B, R, D]

    degenerate = (
        torch.linalg.norm(v, dim=-1) < eps
    ) | (
        torch.linalg.norm(w, dim=-1) < eps
    )
    valid_mask = dihedral_mask & ~degenerate
    angles = torch.where(valid_mask, angles, torch.zeros_like(angles))

    if squeeze_batch:
        return angles[0]
    return angles

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
        use_coords_conditioning = False
        use_dihedrals_conditioning = True
        target_atom_coords = None
        target_dihedrals = None
        dihedral_atom_indices = None
        dihedral_mask = None

        if use_coords_conditioning:
            target_atom_coords = batch.get("target_atom_coords_aligned")
        elif use_dihedrals_conditioning:
            target_dihedrals = batch.get("dihedrals")
            dihedral_atom_indices = batch.get("dihedral_atom_indices")
            dihedral_mask = batch.get("dihedral_mask")
            target_dihedrals[~dihedral_mask] = float("nan")

        use_exendiff = True # keep it like this for now
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
                x0_hat = y_for_grad + (1.0 - t_pad) * velocity_grad # https://gemini.google.com/app/6502a4a95573ee29

                if use_coords_conditioning:
                    # center target into the same frame as the model input / x0_hat
                    target_atom_coords = target_atom_coords.to(y_for_grad)
                    target = center_random_augmentation(
                        target_atom_coords,
                        atom_pad_mask,
                        augmentation=False,
                        centering=True,
                    )
                    residual = (target - x0_hat) * atom_pad_mask_3d # there there should be a **2

                elif use_dihedrals_conditioning:
                    target_dihedrals = target_dihedrals.to(y_for_grad)
                    dihedral_atom_indices = torch.tensor(dihedral_atom_indices).to(
                            device=y_for_grad.device,
                            dtype=torch.long,
                        )
                    dihedral_mask = torch.tensor(dihedral_mask.to(
                            device=y_for_grad.device,
                            dtype=torch.bool,
                        ))

                    pred_dihedrals = A_fn(
                        x0_hat,
                        dihedral_atom_indices,
                        dihedral_mask,
                    )

                    if target_dihedrals.ndim == 4:
                        pred_for_loss = pred_dihedrals.unsqueeze(1)  # [B, 1, R, D]
                        mask_for_loss = dihedral_mask.unsqueeze(1)    # [B, 1, R, D]
                    elif target_dihedrals.ndim == 3:
                        pred_for_loss = pred_dihedrals               # [B, R, D]
                        mask_for_loss = dihedral_mask                # [B, R, D]
                    else:
                        raise ValueError(
                            f"Unsupported target dihedrals shape: {tuple(target_dihedrals.shape)}."
                        )

                    # conditioning_loss = 0.5 * torch.mean((torch.nan_to_num(target_dihedrals) - pred_for_loss) ** 2)
                    # conditioning_loss =  torch.norm((torch.nan_to_num(target_dihedrals) - torch.nan_to_num(pred_for_loss))**2, p=2)


                    # conditioning_loss =  torch.norm(torch.nan_to_num(target_dihedrals) - torch.nan_to_num(pred_for_loss), p=2)**2
                    conditioning_loss =  torch.norm(torch.cos(torch.nan_to_num(target_dihedrals)) - torch.cos(torch.nan_to_num(pred_for_loss)), p=2)**2
                    print((target_dihedrals.nan_to_num()-pred_dihedrals.nan_to_num()).sum())

                if use_coords_conditioning:
                    residual_l2_norm = torch.norm(residual, p=2)**2
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
            score = score -  grad_conditioning # 4 is too much
            self._log_exendiff(t, score_og, score, grad_conditioning_og, grad_conditioning, scale)

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
