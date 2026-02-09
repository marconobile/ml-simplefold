#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule
import hydra
from omegaconf import OmegaConf

from utils.utils import (
    extras,
    create_folders,
    task_wrapper,
)
from utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_trainer,
)
from utils.logging_utils import log_hyperparameters
from utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# Use the new matmul precision API only to avoid mixing legacy TF32 flags.
torch.set_float32_matmul_precision("medium")


@task_wrapper
def train(cfg):
    seed = cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    load_ckpt_path = cfg.get("load_ckpt_path", None)
    ckpt_path = load_ckpt_path

    if load_ckpt_path is not None:
        # load existing ckpt
        log.info(f"Resuming from checkpoint <{cfg.load_ckpt_path}>...")
        model.strict_loading = False

        # manually reset these variables in case of fine-tuning
        model.lddt_weight_schedule = cfg.model.get("lddt_weight_schedule", False)
        model.plddt_training = cfg.model.get("plddt_training", False)

        # reset ESM model to avoid issues in loading FSDP checkpoint
        model.reset_esm(cfg.model.esm_model)

        # Determine if this is a full Lightning checkpoint or weights-only
        ckpt = torch.load(load_ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "pytorch-lightning_version" in ckpt:
            log.info("Detected a PyTorch Lightning checkpoint. Trainer state will be restored.")
        else:
            log.info("Detected a weights-only checkpoint. Loading weights manually.")
            loaded = False
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                try:
                    model.load_state_dict(ckpt["state_dict"], strict=False)
                    loaded = True
                except Exception as e:
                    log.warning(f"Failed to load state_dict from checkpoint: {e}")

            if not loaded:
                # Fall back to EMA-only weights (official release format)
                try:
                    for key in model.model_ema.state_dict().keys():
                        src_key = key.replace("module.", "")
                        if src_key in ckpt:
                            model.model_ema.state_dict()[key].copy_(ckpt[src_key])
                    model.load_state_dict(model.model_ema.state_dict(), strict=False)
                    loaded = True
                except Exception as e:
                    log.warning(f"Failed to load EMA weights from checkpoint: {e}")

            if not loaded:
                raise RuntimeError(
                    "Failed to load weights from checkpoint. "
                    "Expected a Lightning checkpoint or a weights-only dict."
                )

            # We already loaded weights; do not let Lightning try to restore trainer state.
            ckpt_path = None

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    OmegaConf.set_struct(cfg.logger, True)
    loggers = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = instantiate_trainer(
        cfg.trainer, callbacks=callbacks, logger=loggers, plugins=None
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }

    if log:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting training!")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="base_train.yaml")
def submit_run(cfg):
    OmegaConf.resolve(cfg)
    extras(cfg)
    create_folders(cfg)
    train(cfg)
    return


if __name__ == "__main__":
    submit_run()
