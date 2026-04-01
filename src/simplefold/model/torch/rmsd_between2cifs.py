import numpy as np, gemmi

base_path = "/home/nobilm@usi.ch/ml-simplefold/samplings/A2a_exendiff_sweep_mar31/baseline_no_exendiff/predictions_simplefold_100M/a2a_nocappings_sampled_0.cif"
new_path  = "/home/nobilm@usi.ch/ml-simplefold/samplings/A2a_exendiff_repro_best/predictions_simplefold_100M/a2a_nocappings_sampled_0.cif"

def load_atoms(path):
    st = gemmi.read_structure(path)
    model = st[0]
    out = {}
    for ch in model:
        cid = str(ch.name)
        for res in ch:
            resid = int(res.seqid.num)
            icode = str(res.seqid.icode).strip()
            for atom in res:
                if atom.element.name == "H":
                    continue
                key = (cid, resid, icode, str(atom.name).strip().upper())
                out[key] = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
    return out

A = load_atoms(base_path)
B = load_atoms(new_path)
keys = sorted(set(A) & set(B))
X = np.stack([A[k] for k in keys])
Y = np.stack([B[k] for k in keys])

Xc = X - X.mean(0)
Yc = Y - Y.mean(0)
U, _, Vt = np.linalg.svd(Yc.T @ Xc)
R = Vt.T @ U.T
if np.linalg.det(R) < 0:
    Vt[-1, :] *= -1
    R = Vt.T @ U.T
Y_aligned = Yc @ R + X.mean(0)

rmsd = np.sqrt(np.mean(np.sum((X - Y_aligned) ** 2, axis=1)))
print(f"matched_atoms={len(keys)}")
print(f"RMSD_new_vs_baseline={rmsd:.6f}")

