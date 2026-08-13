'''
THIS IS USED TO WRITE THE TEST DATASET, GIVEN A TRAIN DATASET THAT DEFINES THE MAPPING TO GLOBAL CLUSTER IDS
(hardcoded below in the script as path to the train dataset npz file)
write_global_cluster_in_test_data.py \
--input-npz input_test_data_npz_without_global_clusters.npz
--output-npz test_data_with_global_clusters.npz
'''

import argparse
import numpy as np
from pathlib import Path
import subprocess
import sys

# get cmd line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--input-npz")
parser.add_argument("--output-npz")
args = parser.parse_args()
input_npz = args.input_npz
output_npz = args.output_npz

subprocess.run(
    [
        sys.executable,
        str(Path(__file__).resolve().with_name("drop_hs.py")),
        "--input-npz",
        input_npz,
    ],
    check=True,
)
input_npz = str(
    Path(input_npz).resolve().parent / "without_hs" / "backmapping_dataset.npz"
)

# seq = 'SSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSLAAADIAVGVLAIPFAITISTGFCAACHGCLFIACFVLVLTQSSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKNHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRS'

# step 1: load TRAIN data, must be train since onto it it is defined the global clusters mapping
path = "/storage_common/nobilm/backmapping_pots_model/datasets/ANECAG_THEO_INZMA_INACTIVE_merged/without_hs/ANECAG_THEO_INZMA_INACTIVE_merged_with_globalclusters.npz" # reference used to define the mapping to global cluster ids
data = dict(np.load(path))

# atom_resids and res_idx are supposed to be already fixed in train data when executing add_atom_idx_and_glob_cluster_id_per_frame_to_npz.py
assert (
    data['atom_resids'][0] == 0 and 
    data['atom_resids'][-1] == 299 and 
    data['atom_resids'].shape[0] == 2338
)

assert (
    data['res_idx'][0] == 0 and 
    data['res_idx'][-1] == 299 and 
    data['res_idx'].shape[0]==300
)

atom_resids = data['atom_resids'] 
res_idx = list(data['res_idx'])


# step 2: build the mapping to global_id
# create a list where each element is a tuple of (res_idx, residue_cluster_count) for that res_idx, since res_idx is constant across active/inactive/pas - since we are using only 1 protein -
# then we can build the mapping to global_id by iterating through the list residue_cluster_count assigning/creating a global_id for each cluster in that range(residue_cluster_count)
# so to do the mapping of clusters sampled the pots model we need to load up one of the active/inactive/pas npz files, get the res_idx and residue_cluster_counts, build the mapping to global_id, 
# s.t. then we can use that mapping to convert the cluster ids sampled by the pots model to global cluster ids that we can then use for conditioning the folding model @ inference time

l = list(zip(data['res_idx'], data['residue_cluster_counts'])) 

# build the res_idx_to_glob_cluster_ids:
res_idx_to_glob_cluster_ids = {}
global_id = 0
for res_id, clu_count in l:
    for clu_global_idx in range(clu_count):
        res_idx_to_glob_cluster_ids[(res_id, clu_global_idx)] = global_id
        global_id += 1


# build the actual data
res_idx_and_glob_cluster_id = []
for frame in data['residue_cluster_ids']:
    for_this_frame = []
    for res_idx, cluster_id in enumerate(frame):
        glob_cluster_id = res_idx_to_glob_cluster_ids[(res_idx, cluster_id)]
        # for_this_frame.append((res_idx, glob_cluster_id))
        for_this_frame.append(glob_cluster_id)
    res_idx_and_glob_cluster_id.append(for_this_frame)

data['res_idx_and_glob_cluster_id_per_frame'] = np.array(res_idx_and_glob_cluster_id) # (49996, 300)
data['atom_idx_and_glob_cluster_id_per_frame'] = data['res_idx_and_glob_cluster_id_per_frame'][:, atom_resids] # (49996, 2338)    

# load test data 
TEST_DATA = dict(np.load(input_npz))
if 'labels' in TEST_DATA: res_key = 'labels'
else: res_key = 'residue_cluster_ids'
if 'labels' in TEST_DATA and 'residue_cluster_ids' in TEST_DATA: raise ValueError("ERROR: both 'labels' and 'residue_cluster_ids' are present in TEST_DATA, be sure of what to use")

res_idx_and_glob_cluster_id_test_data = []
for frame in TEST_DATA[res_key]:
    for_this_frame = []
    for res_idx, cluster_id in enumerate(frame):
        glob_cluster_id = res_idx_to_glob_cluster_ids[(res_idx, int(cluster_id))]
        for_this_frame.append(glob_cluster_id)
    res_idx_and_glob_cluster_id_test_data.append(for_this_frame)

TEST_DATA["res_idx_and_glob_cluster_id_per_frame"] = np.array(
    res_idx_and_glob_cluster_id_test_data
)    

TEST_DATA["atom_idx_and_glob_cluster_id_per_frame"] = TEST_DATA[
        "res_idx_and_glob_cluster_id_per_frame"
    ][:, atom_resids]

TEST_DATA['og_npz_path'] = input_npz
TEST_DATA['atom_resids'] = data['atom_resids']
TEST_DATA['res_idx'] = data['res_idx']

np.savez_compressed(output_npz, **TEST_DATA)