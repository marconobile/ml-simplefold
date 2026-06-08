# How to: `torch.jit.trace(model, ...)` for Inference

This model does not currently compile with `torch.jit.script(model)`, but the
`FoldingDiT` forward pass can be compiled with `torch.jit.trace(...)` for
inference.

The important constraints are:

- Trace only the `FoldingDiT` model, not ESM preprocessing, data loading,
  postprocessing, or the Python sampler.
- Trace after `prepare_conditioned_batch(...)` has already produced a real
  inference batch with `esm_s`.
- Pass only tensor entries from `batch` to the traced model.
- Treat the trace as topology/shape-specific. For `active_without_hs.npz`, the
  checked shape was `B=1`, `N=2338` atoms, `M=300` tokens.

## Why Trace, Not Script

Direct scripting fails on the current Python model code:

```python
scripted = torch.jit.script(model)
```

The first concrete failure is in `AbsolutePositionEncoding.get_1d_pos_embed`,
where TorchScript cannot prove that the local variable `out` is always defined.
There are additional scripting-hostile patterns in the model path, including
feature dictionaries, optional keys, `**kwargs` through transformer blocks, and
Python shape-dependent branches.

Tracing works because it records the eval-time tensor path for one concrete
input topology.

## Load the Original `last.ckpt`

This is the same checkpoint-loading path used by
`scripts/sample_with_conditioning.py`.

Run from the repository root:

```python
from pathlib import Path
import sys
import torch

REPO_ROOT = Path("/home/nobilm@usi.ch/ml-simplefold")
sys.path.insert(0, str(REPO_ROOT / "src" / "simplefold"))

from scripts.sample_with_conditioning import instantiate_and_load_model

checkpoint_path = Path(
    "/storage_common/nobilm/ml-simplefold/"
    "fine_tune_with_clusters/ft_merged_npz_from_simplefold100M/"
    "checkpoints/last.ckpt"
)
architecture_config = REPO_ROOT / "configs/model/architecture/foldingdit_100M.yaml"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = instantiate_and_load_model(
    architecture_config=architecture_config,
    checkpoint_path=checkpoint_path,
    device=device,
    prefer_ema=True,
    use_mmap=True,
)
model.eval()
```

`prefer_ema=True` loads checkpoint keys with the `model_ema.module.` prefix,
which is the default inference behavior in the evaluation script.

## Trace After Batch Preparation

In `scripts/sample_with_conditioning.py`, the right place to trace
is after this block has run:

```python
batch, structure, record = prepare_conditioned_batch(...)
```

and after the model has been loaded:

```python
model = instantiate_and_load_model(...)
```

Then trace with the real batch:

```python
trace_output_path = Path(
    "/storage_common/nobilm/ml-simplefold/"
    "fine_tune_with_clusters/ft_merged_npz_from_simplefold100M/"
    "checkpoints/foldingdit_active_traced.pt"
)

# TorchScript accepts Dict[str, Tensor]. The full batch also contains entries
# such as "aa_seq" and "record", so remove non-tensor values.
tensor_batch = {
    key: value
    for key, value in batch.items()
    if isinstance(value, torch.Tensor)
}

# Clone the trace inputs because the model currently mutates the cluster-label
# tensor in-place by replacing -1 padding labels with model.pad_idx.
trace_batch = {
    key: value.clone()
    for key, value in tensor_batch.items()
}

example_noise = torch.randn_like(trace_batch["coords"])
example_t = torch.full(
    (example_noise.shape[0],),
    0.5,
    dtype=example_noise.dtype,
    device=example_noise.device,
)

with torch.no_grad():
    traced_model = torch.jit.trace(
        model,
        (example_noise, example_t, trace_batch),
        strict=False,
        check_trace=False,
    )

torch.jit.save(traced_model, trace_output_path)
print(f"Wrote traced model: {trace_output_path}")
```

Expected output shape for `active_without_hs.npz`:

```text
predict_velocity: (1, 2338, 3)
latent:           (1, 300, 768)
```

## Use the Traced Model for Sampling

Load the traced model and pass the tensor-only batch into the existing sampler:

```python
traced_model_path = Path(
    "/storage_common/nobilm/ml-simplefold/"
    "fine_tune_with_clusters/ft_merged_npz_from_simplefold100M/"
    "checkpoints/foldingdit_active_traced.pt"
)

model = torch.jit.load(traced_model_path, map_location=device)
model.eval()

tensor_batch = {
    key: value
    for key, value in batch.items()
    if isinstance(value, torch.Tensor)
}

with torch.no_grad():
    noise = torch.randn_like(tensor_batch["coords"])
    out_dict = sampler.sample(model, flow, noise, tensor_batch)
    out_dict = processor.postprocess(out_dict, tensor_batch)
```

The traced model supports the same keyword call style used by the sampler:

```python
model(
    noised_pos=y,
    t=batched_t,
    feats=tensor_batch,
)
```

## Standalone Smoke Test

This test loads the original checkpoint, builds dummy tensors with the active
NPZ topology, traces the model, and runs one traced forward pass.

```python
from pathlib import Path
import sys
import torch

REPO_ROOT = Path("/home/nobilm@usi.ch/ml-simplefold")
sys.path.insert(0, str(REPO_ROOT / "src" / "simplefold"))

from scripts.sample_with_conditioning import instantiate_and_load_model

B, N, M = 1, 2338, 300
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint_path = Path(
    "/storage_common/nobilm/ml-simplefold/"
    "fine_tune_with_clusters/ft_merged_npz_from_simplefold100M/"
    "checkpoints/last.ckpt"
)
architecture_config = REPO_ROOT / "configs/model/architecture/foldingdit_100M.yaml"


def make_feats(device: torch.device) -> dict[str, torch.Tensor]:
    atom_to_token = torch.zeros(B, N, M, device=device)
    atom_to_token[
        0,
        torch.arange(N, device=device),
        torch.arange(N, device=device) % M,
    ] = 1.0

    return {
        "ref_pos": torch.randn(B, N, 3, device=device),
        "mol_type": torch.zeros(B, M, dtype=torch.long, device=device),
        "atom_to_token": atom_to_token,
        "atom_to_token_idx": torch.arange(N, device=device).remainder(M).view(B, N),
        "ref_space_uid": torch.zeros(B, N, dtype=torch.long, device=device),
        "atom_idx_and_glob_cluster_id_per_frame": torch.full(
            (B, N),
            -1,
            dtype=torch.long,
            device=device,
        ),
        "max_num_tokens": torch.tensor([M], dtype=torch.long, device=device),
        "res_type": torch.zeros(B, M, 33, device=device),
        "pocket_feature": torch.zeros(B, M, 4, device=device),
        "ref_charge": torch.zeros(B, N, device=device),
        "atom_pad_mask": torch.ones(B, N, dtype=torch.bool, device=device),
        "ref_element": torch.zeros(B, N, 128, device=device),
        "ref_atom_name_chars": torch.zeros(B, N, 4, 64, device=device),
        "residue_index": torch.arange(M, device=device).view(B, M),
        "entity_id": torch.zeros(B, M, dtype=torch.long, device=device),
        "asym_id": torch.zeros(B, M, dtype=torch.long, device=device),
        "sym_id": torch.zeros(B, M, dtype=torch.long, device=device),
        "esm_s": torch.zeros(B, M, 37, 2560, device=device),
    }


model = instantiate_and_load_model(
    architecture_config=architecture_config,
    checkpoint_path=checkpoint_path,
    device=device,
    prefer_ema=True,
    use_mmap=True,
)
model.eval()

example_noise = torch.randn(B, N, 3, device=device)
example_t = torch.tensor([0.5], device=device)
example_feats = make_feats(device)

with torch.no_grad():
    eager_out = model(example_noise, example_t, example_feats)
    print({key: tuple(value.shape) for key, value in eager_out.items()})

    traced_model = torch.jit.trace(
        model,
        (example_noise, example_t, make_feats(device)),
        strict=False,
        check_trace=False,
    )

    traced_out = traced_model(example_noise, example_t, make_feats(device))
    print({key: tuple(value.shape) for key, value in traced_out.items()})
```

## Caveats

- Retrace if atom count, token count, padding shape, or ESM model shape changes.
- The Python sampler loop is still Python. This trace only replaces calls to
  `FoldingDiT.forward(...)` inside each Euler-Maruyama step.
- Keep `model.eval()` and `torch.no_grad()` for inference tracing and sampling.
- `check_trace=False` is used because sampling inputs include stochastic paths
  and the model contains shape-dependent Python branches that produce expected
  tracer warnings.
