<!-- If you want to use MLX backend on Apple silicon:
```
pip install mlx==0.28.0
pip install git+https://github.com/facebookresearch/esm.git
pip install rich fairscale tensorboard
``` -->

<!-- ## Example

We provide a jupyter notebook [`sample.ipynb`](sample.ipynb) to predict protein structures from example protein sequences. -->

<!--

## Evaluation

We provide predicted structures from SimpleFold of different model sizes:
```
https://ml-site.cdn-apple.com/models/simplefold/cameo22_predictions.zip # predicted structures of CAMEO22
https://ml-site.cdn-apple.com/models/simplefold/casp14_predictions.zip  # predicted structures of CASP14
https://ml-site.cdn-apple.com/models/simplefold/apo_predictions.zip     # predicted structures of Apo
https://ml-site.cdn-apple.com/models/simplefold/codnas_predictions.zip  # predicted structures of Fold-switch (CoDNaS)
```
We use the docker image of [openstructure](https://git.scicore.unibas.ch/schwede/openstructure/) 2.9.1 to evaluate generated structures for folding tasks (i.e., CASP14/CAMEO22). Once having the docker image enabled, you can run evaluation via:
```
python src/simplefold/evaluation/analyze_folding.py \
    --data_dir [PATH_TO_TARGET_MMCIF] \
    --sample_dir [PATH_TO_PREDICTED_MMCIF] \
    --out_dir [PATH_TO_OUTPUT] \
    --max-workers [NUMBER_OF_WORKERS]
```
To evaluate results of two-state prediction (i.e., Apo/CoDNaS), one need to compile the [TMsore](https://zhanggroup.org/TM-score/TMscore.cpp) and then run evaluation via:
```
python src/simplefold/evaluation/analyze_two_state.py \
    --data_dir [PATH_TO_TARGET_DATA_DIRECTORY] \
    --sample_dir [PATH_TO_PREDICTED_PDB] \
    --tm_bin [PATH_TO_TMscore_BINARY] \
    --task apo \ # choose from apo and codnas
    --nsample 5
``` -->

# Data preparation

## Input expected:
1) processed target: data required (*.npz + *.json)
2) tokenized dataset: data required (*.pkl + *.json + manifest.json)

### Directory layout (see train_datamodule.py):
We assume the training data is stored in with the following structure:

- target_dir_for_dataset_A/
    - structures/ ; created by process_mmcif.py
        - {record_id}.npz # i.e. filename.npz; created by process_mmcif.py

- tokenized_dir_for_dataset_A/
    - tokens/
        - {record_id}.pkl # i.e. filename.pkl
    - records/ ; created by process_mmcif.py
        - {record_id}.json # i.e. filename.json ; created by process_mmcif.py
    - manifest.json # Groups records; created by process_mmcif.py

- target_dir_for_dataset_B/
    - ...
- tokenized_dir_for_dataset_B/
    - ...
- ...

<!-- #### Training targets

SimpleFold is trained on joint datasets including experimental structures from [PDB](https://www.rcsb.org/), as well as distilled predictions from [AFDB SwissProt](https://alphafold.ebi.ac.uk/download#swissprot-section) and [AFESM](https://afesm.foldseek.com/). Target lists of filtered SwissProt and AFESM targets thta are used in our training can be found:
```
https://ml-site.cdn-apple.com/models/simplefold/swissprot_list.csv # list of filted SwissProt (~270K targets)
https://ml-site.cdn-apple.com/models/simplefold/afesm_list.csv # list of filted AFESM targets (~1.9M targets)
https://ml-site.cdn-apple.com/models/simplefold/afesme_dict.json # list of filted extended AFESM (AFESM-E) (~8.6M targets)
```
In `afesme_dict.json`, the data is stored in the following structure:
```
{
    cluster 1 ID: {"members": [protein 1 ID, protein 2 ID, ...]},
    cluster 2 ID: {"members": [protein 1 ID, protein 2 ID, ...]},
    ...
}
```

Of course, one can use own customized datasets to train or tune SimpleFold models. Instructions below list how to process the dataset for SimpleFold training. -->

#### Process mmcif structures

`process_mmcif.py` expects a local directory of mmCIF files (e.g., `*.cif` or `*.cif.gz`). If your `--data_dir` is empty, it will process `0` entries.

For a quick smoke test, you can download a single PDB entry in mmCIF format:
```
mkdir -p data/mmcif
wget -O data/mmcif/1ubq.cif.gz https://files.rcsb.org/download/1UBQ.cif.gz
```
<!-- To download the full PDB mmCIF archive, use the RCSB rsync mirror (this is large):
```
rsync -rlpt -z --delete rsync.rcsb.org::ftp-data/structures/divided/mmCIF/ data/mmcif/
``` -->

To process downloaded mmcif files, you need [Redis](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/) installed and launch the Redis server:
```
wget https://boltz1.s3.us-east-2.amazonaws.com/ccd.rdb
redis-server --dbfilename ccd.rdb --port 7777
```
Note: run `redis-server` from the directory that contains `ccd.rdb`, or pass `--dir /path/to/ccd_dir` so Redis can load the database.
You can then process mmcif files to input format for SimpleFold:
```
python src/simplefold/process_mmcif.py \
    --data_dir [MMCIF_DIR]   # directory of mmcif files
    --out_dir [OUTPUT_DIR]   # directory of processed targets
    --use-assembly
```

If you already have structures as PDB files (e.g., `*.pdb` / `*.pdb.gz`), you can produce the same `structures/`, `records/`, and `manifest.json` outputs via:
```
python src/simplefold/process_pdb.py \
    --data_dir [PDB_DIR] \
    --out_dir [OUTPUT_DIR] \
    --use-assembly
```
# Example data preparation pipeline:
```
redis-server --dbfilename ccd.rdb --port 7777  # Note: run `redis-server` from the directory that contains `ccd.rdb`, or pass `--dir /path/to/ccd_dir` so Redis can load the database.
python src/simplefold/process_pdb.py --data_dir /home/nobilm@usi.ch/ml-simplefold/data/pdb_for_train_test --out_dir /home/nobilm@usi.ch/ml-simplefold/training_data --use-assembly
python src/simplefold/process_structure.py --target_dir /storage_common/nobilm/ml-simplefold/training_data --token_dir /storage_common/nobilm/ml-simplefold/training_data
```
-> output saved in: --out_dir /home/nobilm@usi.ch/ml-simplefold/training_data


<!-- ```
python src/simplefold/process_pdb.py --data_dir data/pdb_inapo --out_dir data/target_inapo --use-assembly
```
-> output: /home/nobilm@usi.ch/ml-simplefold/data/target_inapo
or:
```
``` -->
<!--
To further tokenize the processed structures:
```
python src/simplefold/process_structure.py \
--target_dir [TARGET_DIR]   # directory of processed targets \
--token_dir [TOKEN_DIR]   # directory of tokenized data # do the same of target_dir
```

##### step 2:
```
python src/simplefold/process_structure.py --target_dir /storage_common/nobilm/ml-simplefold/training_data --token_dir /storage_common/nobilm/ml-simplefold/training_data
``` -->


## Training

The configuration of model is based on [`Hydra`](https://hydra.cc/docs/intro/). An example training configuration can be found in `configs/experiment/train`. To change dataset and model settings, one can refer to config files in `configs/data` and `configs/model`.
<!-- To initiate SimpleFold training:
```
python train experiment=train
```
To train SimpleFold with FSDP strategy:
```
python train_fsdp.py experiment=train_fsdp
``` -->

### To train from scratch:
```
CUDA_VISIBLE_DEVICES=0 python src/simplefold/train.py experiment=train \
  hydra.job.name=inapo_2ep_gpu0 \
  paths.output_dir=/home/nobilm@usi.ch/ml-simplefold/artifacts/inapo_2ep_gpu0 \
  trainer.accelerator=gpu trainer.devices=1 +trainer.max_epochs=2
```

### To train from checkpoint:
```
CUDA_VISIBLE_DEVICES=0 python src/simplefold/train.py experiment=train   hydra.job.name=inapo_1000ep_gpu0   paths.output_dir=/storage_common/nobilm/ml-simplefold/artifacts/inapo_1000ep_gpu0   +load_ckpt_path=/home/nobilm@usi.ch/ml-simplefold/artifacts/simplefold_100M.ckpt   trainer.accelerator=gpu trainer.devices=1 +trainer.max_epochs=10   callbacks.model_checkpoint.dirpath=/storage_common/nobilm/ml-simplefold/artifacts/inapo_1000ep_gpu0/checkpoints   "callbacks.model_checkpoint.filename='sf100M-ep{epoch:04d}-step{step:08d}'"   +callbacks.model_checkpoint.every_n_epochs=3   callbacks.model_checkpoint.save_on_train_epoch_end=True   callbacks.model_checkpoint.every_n_train_steps=null
```

#### Tensorboard:
cd /run_name/tensorboard; run tensorboard --logdir=.

<!-- ## Citation
If you found this code useful, please cite the following paper:
```
@article{simplefold,
  title={SimpleFold: Folding Proteins is Simpler than You Think},
  author={Wang, Yuyang and Lu, Jiarui and Jaitly, Navdeep and Susskind, Josh and Bautista, Miguel Angel},
  journal={arXiv preprint arXiv:2509.18480},
  year={2025}
}
```

## Acknowledgements
Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details.

## License
Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.
 -->


# Inference

Once you have `simplefold` package installed, you can predict the protein structure from target fasta file(s) via the following command line. We provide support for both [PyTorch](https://pytorch.org/) and [MLX](https://mlx-framework.org/) (recommended for Apple hardware) backends in inference.
```
simplefold \
    --simplefold_model simplefold_100M \  # specify folding model in simplefold_100M/360M/700M/1.1B/1.6B/3B
    --num_steps 500 --tau 0.01 \          # specify inference setting
    --nsample_per_protein 1 \             # number of generated conformers per target
    --plddt \                             # output pLDDT
    --fasta_path [FASTA_PATH] \           # path to the target fasta directory or file
    --output_dir [OUTPUT_DIR] \           # path to the output directory
    --backend [mlx, torch]                # choose from MLX and PyTorch for inference backend
```
### or cmd to execute cli:
```
python /home/nobilm@usi.ch/ml-simplefold/src/simplefold/cli.py --output_dir /home/nobilm@usi.ch/ml-simplefold/A2a_via_cli --fasta_path /home/nobilm@usi.ch/ml-simplefold/a2a.fasta
```

# Installation

To install `simplefold` package from github repository, run
```
git clone https://github.com/apple/ml-simplefold.git
cd ml-simplefold
conda create -n simplefold python=3.10
conda activate simplefold
python -m pip install -U pip build; pip install -e .
```


# Final Dataset for Backmapping:
ACTIVE: /storage_common/angiod/phase-data/projects/a2a/systems/a2a/clusters/cb3c3cae-5316-47db-8fbb-0567d5f0f75b/samples/e98051c1-744f-4522-bafd-2bfdeea9788b/backmapping_dataset.npz
INACTIVE: /storage_common/angiod/phase-data/projects/a2a/systems/a2a/clusters/cb3c3cae-5316-47db-8fbb-0567d5f0f75b/samples/7cd3f1a1-3b15-4de3-b1e3-5d2d1176b5fe/backmapping_dataset.npz
PAS: /storage_common/angiod/phase-data/projects/a2a/systems/a2a/clusters/cb3c3cae-5316-47db-8fbb-0567d5f0f75b/samples/f121244e-b111-4e84-a67e-51f677718282/backmapping_dataset.npz


trajectory (49996, 4746, 3) # xyz
atom_resids (4746,) # this starts from 1 # first el is capping to be removed
atom_names (4746,)
atom_residue_index (4746,) # this starts from 0
residue_keys (300,)
residue_cluster_ids (49996, 300)
residue_cluster_counts (300,)
frame_indices (49996,)
frame_state_ids (49996,)
state_id ()
sample_id ()
dihedrals (49996, 300, 5)
dihedral_keys (5,)


python drop_hs.py --input-npz /storage_common/nobilm/backmapping_pots_model/datasets/active/with_hs/backmapping_dataset.npz
out: /storage_common/nobilm/backmapping_pots_model/datasets/active/without_hs/backmapping_dataset.npz

python drop_hs.py --input-npz /storage_common/nobilm/backmapping_pots_model/datasets/inactive/with_hs/backmapping_dataset.npz
out: /storage_common/nobilm/backmapping_pots_model/datasets/inactive/without_hs/backmapping_dataset.npz

python drop_hs.py --input-npz /storage_common/nobilm/backmapping_pots_model/datasets/pas/with_hs/backmapping_dataset.npz
out: /storage_common/nobilm/backmapping_pots_model/datasets/pas/without_hs/backmapping_dataset.npz

https://rest.uniprot.org/uniprotkb/P29274.fasta

python /home/nobilm@usi.ch/ml-simplefold/src/simplefold/cli.py \
  --backend torch \
  --fasta_path /home/nobilm@usi.ch/ml-simplefold/a2a_nocappings.fasta \
  --output_dir /home/nobilm@usi.ch/ml-simplefold/A2a_via_cli_test_exendiff \
  --target_conditioning_npz /storage_common/nobilm/backmapping_pots_model/datasets/active/without_hs/backmapping_dataset.npz \
  --target_frame_idx 0

python scripts/cif_npz_atomwise_rmsd.py  \
--cif-path /home/nobilm@usi.ch/ml-simplefold/dbg_output_1000/predictions_simplefold_100M/a2a_nocappings_sampled_0.cif \
--npz-path /storage_common/nobilm/backmapping_pots_model/datasets/inactive/whs/without_hs/backmapping_dataset.npz  \
--frame-index 0 \
--per-atom-out /home/nobilm@usi.ch/ml-simplefold/dbg_output_1000/predictions_simplefold_100M/a2a_nocappings_sampled_0_atomwise_rmsd.csv
