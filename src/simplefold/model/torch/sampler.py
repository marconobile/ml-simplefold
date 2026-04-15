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

# def generate_pattern(blocks=100, n=1):
#     result = list(range(0, n + 1))  # first block: 0..n

#     for k in range(blocks):
#         start = k * n
#         end = (k + 2) * n
#         result.extend(range(start, end + 1))
#     return result

# def generate_pattern_v2(N = 500):
#     timesteps = []
#     for i in range(N):
#         timesteps.append(i)
#         timesteps.append(i)
#     return timesteps


def generate_pattern(N = 500):
    timesteps = []
    for i in range(N):
        timesteps.append(i)
        timesteps.append(i)

    last_val = timesteps[-1]+1
    for i in range(int(1.2*N)):
        timesteps.append(last_val+i)
    return timesteps

# def generate_pattern(blocks=500, n=1): # v5
#     result = list(range(0, n + 1))  # first block: 0..n

#     for k in range(blocks):
#         start = k * n
#         end = (k + 2) * n
#         result.extend(range(start, end + 1))

#     last_val = result[-1]+1
#     for i in range(blocks):
#         result.append(last_val+i)
#     return result




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
        trajectory_coord_scale=1.0,
    ):
        self.num_timesteps = num_timesteps
        self.log_timesteps = log_timesteps
        self.t_start = t_start
        self.tau = tau
        self.w_cutoff = w_cutoff
        self.guidance_scale = guidance_scale
        self.conditioning_key = conditioning_key
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.trajectory_coord_scale = float(trajectory_coord_scale)
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
    def _decode_atom_name_code(code_row):
        chars = [chr(int(c) + 32) for c in code_row if int(c) != 0]
        return "".join(chars).strip().upper()

    @staticmethod
    def _atom_name_to_xyz_symbol(atom_name):
        atom_name = str(atom_name).strip().upper()
        letters = [char for char in atom_name if char.isalpha()]
        if len(letters) == 0:
            return "X"
        return letters[0]

    @staticmethod
    def _sanitize_filename_component(value):
        text = str(value).strip()
        if len(text) == 0:
            return "diffusion"
        safe = "".join(
            ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
            for ch in text
        )
        safe = safe.strip("._")
        return safe if len(safe) > 0 else "diffusion"

    def _trajectory_prefix_from_batch(self, batch):
        if not isinstance(batch, dict):
            return "diffusion"

        record = batch.get("record", None)
        record_id = None
        if isinstance(record, dict):
            record_id = record.get("id", None)
        elif isinstance(record, (list, tuple)) and len(record) > 0:
            first = record[0]
            if isinstance(first, dict):
                record_id = first.get("id", None)
            elif hasattr(first, "id"):
                record_id = getattr(first, "id")
        elif hasattr(record, "id"):
            record_id = getattr(record, "id")

        if record_id is None:
            return "diffusion"
        return self._sanitize_filename_component(record_id)

    def _extract_batch_atom_symbols(self, batch, atom_pad_mask):
        batch_size = atom_pad_mask.shape[0]
        default_symbols = [
            np.asarray(["X"] * int(atom_pad_mask[b].sum()), dtype=object)
            for b in range(batch_size)
        ]

        ref_atom_name_chars = batch.get("ref_atom_name_chars", None)
        if ref_atom_name_chars is None or not torch.is_tensor(ref_atom_name_chars):
            return default_symbols

        atom_name_codes = torch.argmax(
            ref_atom_name_chars,
            dim=-1,
        ).detach().cpu().numpy()

        symbols_per_batch = []
        for b in range(batch_size):
            encoded_atom_names = atom_name_codes[b][atom_pad_mask[b]]
            decoded_atom_names = [
                self._decode_atom_name_code(code_row)
                for code_row in encoded_atom_names
            ]
            atom_symbols = [
                self._atom_name_to_xyz_symbol(atom_name)
                for atom_name in decoded_atom_names
            ]
            symbols_per_batch.append(np.asarray(atom_symbols, dtype=object))
        return symbols_per_batch

    def _initialize_xyz_trajectory_writers(self, batch):
        if self.output_dir is None or not isinstance(batch, dict):
            return []

        atom_pad_mask = batch.get("atom_pad_mask", None)
        if atom_pad_mask is None or not torch.is_tensor(atom_pad_mask):
            return []

        atom_pad_mask = atom_pad_mask.detach().cpu().numpy() > 0.5
        batch_size = atom_pad_mask.shape[0]
        atom_symbols_per_batch = self._extract_batch_atom_symbols(batch, atom_pad_mask)
        prefix = self._trajectory_prefix_from_batch(batch)

        writers = []
        for b in range(batch_size):
            xyz_path = self.output_dir / f"{prefix}_sampled_{b}_denoising_trajectory.xyz"
            xyz_path.parent.mkdir(parents=True, exist_ok=True)
            writers.append(
                {
                    "path": xyz_path,
                    "handle": xyz_path.open("w"),
                    "atom_mask": atom_pad_mask[b],
                    "atom_symbols": atom_symbols_per_batch[b],
                }
            )
        return writers

    def _write_xyz_trajectory_frame(self, writers, y_sampled, step_idx, t, t_next):
        if len(writers) == 0:
            return
        if not torch.is_tensor(y_sampled):
            return

        y_np = y_sampled.detach().cpu().float().numpy()
        y_np = y_np * float(self.trajectory_coord_scale)
        for b, writer in enumerate(writers):
            atom_mask = writer["atom_mask"]
            atom_symbols = writer["atom_symbols"]
            coords = y_np[b][atom_mask]
            if len(atom_symbols) != coords.shape[0]:
                atom_symbols = np.asarray(["X"] * coords.shape[0], dtype=object)

            handle = writer["handle"]
            handle.write(f"{coords.shape[0]}\n")
            handle.write(f"step={step_idx} t={t:.6f} t_next={t_next:.6f}\n")
            for atom_symbol, xyz in zip(atom_symbols, coords):
                handle.write(
                    f"{atom_symbol:>2} {float(xyz[0]): .6f} {float(xyz[1]): .6f} {float(xyz[2]): .6f}\n"
                )

    def _finalize_xyz_trajectory_writers(self, writers):
        for writer in writers:
            writer["handle"].close()
            print(f"Saved diffusion trajectory XYZ to: {writer['path']}")

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
            ax.set_ylabel("wrapped error [rad]")
            if np.any(finite_idx):
                y_finite = y_vals[finite_idx]
                y_min = float(np.min(y_finite))
                y_max = float(np.max(y_finite))
                if y_max > y_min:
                    ax.set_ylim(y_min, y_max)
                else:
                    # Avoid singular axis when all values are identical.
                    pad = max(1e-6, 1e-3 * max(abs(y_min), 1.0))
                    ax.set_ylim(
                        max(0.0, y_min - pad),
                        min(float(np.pi), y_max + pad),
                    )
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
        i,
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
        use_coords_conditioning = False #True and i%6==0
        use_dihedrals_conditioning = True
        target_atom_coords = None
        target_dihedrals = None
        dihedral_atom_indices = None
        dihedral_mask = None

        target_atom_coords = batch.get("target_atom_coords_aligned")
        target_dihedrals = batch.get("dihedrals")
        dihedral_atom_indices = batch.get("dihedral_atom_indices")
        dihedral_mask = batch.get("dihedral_mask")

        use_exendiff = False # keep it like this for now
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
                    residual_l2_norm = torch.norm(residual, p=2)**2
                    conditioning_loss = 0.5 * residual_l2_norm

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

            score  = score - 2.5 * grad_conditioning

            # if i%5 < 4:
            #     score = - grad_conditioning
            # else:
            #     score = score - grad_conditioning

            if use_coords_conditioning:
                self._log_exendiff(t, score_og, score, grad_conditioning_og, grad_conditioning, scale, {"conditioning_loss mse": (conditioning_loss).sum().item()})
            elif use_dihedrals_conditioning:
                self._log_exendiff(t, score_og, score, grad_conditioning_og, grad_conditioning, scale, {"conditioning_loss dihedrals": (target_dihedrals.nan_to_num()-pred_dihedrals.nan_to_num()).sum().item()})

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
        trajectory_writers = self._initialize_xyz_trajectory_writers(feats)
        try:
            if len(trajectory_writers) > 0:
                t0 = float(steps[0].detach().cpu().item())
                self._write_xyz_trajectory_frame(
                    trajectory_writers,
                    y_sampled,
                    step_idx=0,
                    t=t0,
                    t_next=t0,
                )

            l = generate_pattern()

            #####################################################
            #!#### og:
            # for i in tqdm(
            #     range(sampling_timesteps),
            #     desc="Sampling",
            #     total=sampling_timesteps,
            # ):
            #!#### better than below?:
            for i in tqdm(
                l,
                desc="Sampling",
                total=len(l),
            ):

            #!#### droppable?
            # for idx in tqdm(
            #     range(len(l)),
            #     desc="Sampling",
            #     total=len(l),
            # ):
            #     if idx+1 == len(l):
            #         break

            #     if l[idx] > l[idx+1]:
            #         continue

            #     i = l[idx]

            #####################################################

                t = steps[i]
                t_next = steps[i + 1]

                y_sampled = self.euler_maruyama_step(
                    model_fn,
                    flow,
                    y_sampled,
                    t,
                    t_next,
                    feats,
                    i,
                )

                if len(trajectory_writers) > 0:
                    self._write_xyz_trajectory_frame(
                        trajectory_writers,
                        y_sampled,
                        step_idx=i + 1,
                        t=float(t.detach().cpu().item()),
                        t_next=float(t_next.detach().cpu().item()),
                    )
        finally:
            self._finalize_xyz_trajectory_writers(trajectory_writers)

        self._save_dihedral_error_plot()
        return {
            "denoised_coords": y_sampled
        }
