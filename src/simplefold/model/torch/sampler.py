#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

from datetime import datetime
from pathlib import Path

import numpy as np
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
        output_dir=None,
    ):
        self.num_timesteps = num_timesteps
        self.log_timesteps = log_timesteps
        self.t_start = t_start
        self.tau = tau
        self.w_cutoff = w_cutoff
        self.guidance_scale = guidance_scale
        self.conditioning_key = conditioning_key
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._dihedral_error_history = []
        self._dihedral_time_history = []
        self._dihedral_valid_count_history = []
        self._dihedral_names = ("phi", "psi", "omega", "chi1", "chi2")

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.log_timesteps:
            t = 1.0 - torch.logspace(-2, 0, self.num_timesteps + 1).flip(0)
            t = t - torch.min(t)
            t = t / torch.max(t)
            self.steps = t.clamp(min=self.t_start, max=1.0)
        else:
            self.steps = torch.linspace(
                self.t_start, 1.0, steps=self.num_timesteps + 1
            )

    @staticmethod
    def _mean_abs_wrapped_dihedral_error(pred_dihedrals, target_dihedrals, dihedral_mask):
        angle_delta = torch.atan2(
            torch.sin(pred_dihedrals - target_dihedrals),
            torch.cos(pred_dihedrals - target_dihedrals),
        ).abs()

        valid_mask = (
            dihedral_mask
            & torch.isfinite(target_dihedrals)
            & torch.isfinite(pred_dihedrals)
        )
        valid_mask_float = valid_mask.to(dtype=angle_delta.dtype)
        masked_angle_delta = torch.where(
            valid_mask,
            angle_delta,
            torch.zeros_like(angle_delta),
        )

        reduce_dims = tuple(range(angle_delta.ndim - 1))
        numerator = masked_angle_delta.sum(dim=reduce_dims)
        denominator = valid_mask_float.sum(dim=reduce_dims)
        mean_error = numerator / denominator.clamp_min(1.0)
        mean_error = torch.where(
            denominator > 0,
            mean_error,
            torch.full_like(mean_error, float("nan")),
        )
        return mean_error.detach().cpu(), denominator.detach().cpu()

    def _save_dihedral_error_plot(self):
        if self.output_dir is None or len(self._dihedral_error_history) == 0:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not available; skipping dihedral error plotting.")
            return None

        history = torch.stack(self._dihedral_error_history, dim=0).numpy()
        num_steps, num_dihedrals = history.shape
        valid_counts = None
        if len(self._dihedral_valid_count_history) == num_steps:
            valid_counts = torch.stack(self._dihedral_valid_count_history, dim=0).numpy()

        ncols = min(3, num_dihedrals)
        nrows = (num_dihedrals + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.8 * ncols, 3.6 * nrows),
            sharex=True,
        )
        axes = np.atleast_1d(axes).reshape(-1)

        if len(self._dihedral_time_history) == num_steps:
            step_ids = np.asarray(self._dihedral_time_history, dtype=np.float64)
        else:
            step_ids = np.arange(num_steps, dtype=np.float64)
        for dihedral_idx in range(num_dihedrals):
            ax = axes[dihedral_idx]
            y_vals = history[:, dihedral_idx]
            finite_idx = np.isfinite(y_vals)
            channel_count = (
                int(np.nanmax(valid_counts[:, dihedral_idx]))
                if valid_counts is not None
                else None
            )
            if channel_count is not None:
                count_suffix = f" (n={channel_count})"
            else:
                count_suffix = ""
            if np.any(finite_idx):
                ax.plot(step_ids[finite_idx], y_vals[finite_idx], linewidth=1.6)
            dihedral_name = (
                self._dihedral_names[dihedral_idx]
                if dihedral_idx < len(self._dihedral_names)
                else f"dihedral_{dihedral_idx}"
            )
            ax.set_title(f"{dihedral_name}{count_suffix}")
            if not np.any(finite_idx):
                ax.text(
                    0.5,
                    0.5,
                    "no valid data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            ax.set_xlabel("time t")
            ax.set_ylabel("error [rad]")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

        for ax in axes[num_dihedrals:]:
            ax.set_visible(False)

        fig.suptitle("Per-dihedral absolute wrapped error over diffusion time", fontsize=12)
        fig.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = self.output_dir / f"dihedral_error_over_time_{timestamp}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved dihedral error plot to: {out_path}")
        if valid_counts is not None:
            max_counts = np.max(valid_counts, axis=0).astype(int).tolist()
            print(
                "Per-dihedral valid counts used for error plotting: "
                f"phi={max_counts[0]}, psi={max_counts[1]}, omega={max_counts[2]}, "
                f"chi1={max_counts[3]}, chi2={max_counts[4]}"
            )
        return out_path

    def _log_exendiff(self, t, score_og, score_new, grad_conditioning_og, grad_conditioning_new, scale, kwargs):

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
                f"scale: {scale:.8f}\n",
                # ", ".join(f"{k}: {v}" for k, v in kwargs.items()) + "\n"
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

        use_exendiff = True # keep it like this for now
        step_dihedral_error = None
        step_dihedral_valid_count = None
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

                    step_dihedral_error, step_dihedral_valid_count = self._mean_abs_wrapped_dihedral_error(
                        pred_for_loss,
                        target_dihedrals,
                        mask_for_loss,
                    )

                    # conditioning_loss = 0.5 * torch.mean((torch.nan_to_num(target_dihedrals) - pred_for_loss) ** 2)
                    # conditioning_loss =  torch.norm((torch.nan_to_num(target_dihedrals) - torch.nan_to_num(pred_for_loss))**2, p=2)
                    # conditioning_loss =  torch.norm(torch.nan_to_num(target_dihedrals) - torch.nan_to_num(pred_for_loss), p=2)**2
                    valid_for_loss = mask_for_loss & torch.isfinite(target_dihedrals)
                    cos_residual = torch.cos(target_dihedrals) - torch.cos(pred_for_loss)
                    cos_residual = torch.where(
                        valid_for_loss,
                        cos_residual,
                        torch.zeros_like(cos_residual),
                    )
                    conditioning_loss = torch.norm(cos_residual, p=2) ** 2

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
            score = score - 2.5 * grad_conditioning # 4 is too much
            # score = score - grad_conditioning # 4 is too much

            self._log_exendiff(t, score_og, score, grad_conditioning_og, grad_conditioning, scale, {"conditioning_loss": (target_dihedrals.nan_to_num()-pred_dihedrals.nan_to_num()).sum().item()})

        #* End of exendiff conditioning.

        diff_coeff = self.diffusion_coefficient(t)
        drift = velocity + diff_coeff * score
        mean_y = y + drift * dt
        y_sample = mean_y + torch.sqrt(2.0 * dt * diff_coeff * self.tau) * eps

        if step_dihedral_error is not None:
            self._dihedral_error_history.append(step_dihedral_error)
            self._dihedral_time_history.append(float(t.detach().cpu().item()))
            if step_dihedral_valid_count is not None:
                self._dihedral_valid_count_history.append(step_dihedral_valid_count)

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
        self._dihedral_error_history = []
        self._dihedral_time_history = []
        self._dihedral_valid_count_history = []
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

        self._save_dihedral_error_plot()
        return {
            "denoised_coords": y_sampled
        }
