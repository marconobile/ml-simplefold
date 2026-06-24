import argparse

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--input-npz")
parser.add_argument("--output-npz")
args = parser.parse_args()

input_npz = args.input_npz
output_npz = args.output_npz

data = dict(np.load(input_npz))

# fix atom_resids by shifting it down by 2 since it was starting at 2 and we like at 0 s.t. avoid any mistake when indexing
new_atom_resids = data['atom_resids'] - 2; new_atom_resids
data['atom_resids'] = new_atom_resids; new_atom_resids
seq = 'SSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSLAAADIAVGVLAIPFAITISTGFCAACHGCLFIACFVLVLTQSSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKNHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRS'
res_idx = list(set(data['atom_residue_index'].tolist()))
res_idx.sort()
data['res_idx'] = res_idx


## build the mapping to global_id
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

# res_idx_to_glob_cluster_ids[(299, 17)] # 1682; 1682*768 = nn.embeddings 1'291'776

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
data['atom_idx_and_glob_cluster_id_per_frame'] = data['res_idx_and_glob_cluster_id_per_frame'][:, new_atom_resids] # (49996, 2338)    

np.savez_compressed(output_npz, **data)

data_new = dict(np.load(output_npz))
for k,v in data_new.items(): print(k, v.shape)
