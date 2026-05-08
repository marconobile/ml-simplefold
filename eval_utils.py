import os
import sys
import numpy as np
from Bio import PDB


def flip_pdb_coordinates(input_path, output_path):
    """
    Reads a PDB file, multiplies all atom coordinates by -1.0, 
    and saves the result to a new file.
    """
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Loading structure from: {input_path}")
    
    # Initialize PDB parser and structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', input_path)

    # Iterate through all atoms in the structure and multiply coords by -1.0
    count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # Update the coordinates in place
                    new_coords = atom.get_coord() * -1.0
                    atom.set_coord(new_coords)
                    count += 1

    print(f"Flipped coordinates for {count} atoms.")

    # Save the modified structure
    io = PDB.PDBIO()
    io.set_structure(structure)
    
    try:
        io.save(output_path)
        print(f"Successfully saved flipped structure to: {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")


def calculate_improper_dihedral(p1, p2, p3, p4):
    """
    Calculates the improper dihedral angle between four points.
    Used for the CORN rule (N, CA, C, CB).
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(b2, b3)
    n2 /= np.linalg.norm(n2)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    
    return np.degrees(np.arctan2(y, x))

def detect_chirality(pdb_path):
    """
    Detects L or D chirality for residues in a PDB file using the CORN rule.
    CORN rule: Looking from HA to CA, the groups CO, R (side chain), and N 
    should appear clockwise for L-amino acids.
    Alternatively, the improper dihedral of N-CA-C-CB is positive for L and negative for D.
    """
    if not os.path.exists(pdb_path):
        print(f"Error: File {pdb_path} not found.")
        return

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    results = []

    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip heteroatoms and Glycine (Glycine is achiral)
                if residue.id[0] != " " or residue.get_resname() == "GLY":
                    continue
                
                try:
                    n = residue['N'].get_vector().get_array()
                    ca = residue['CA'].get_vector().get_array()
                    c = residue['C'].get_vector().get_array()
                    cb = residue['CB'].get_vector().get_array()
                    
                    # Calculate improper dihedral N-CA-C-CB
                    angle = calculate_improper_dihedral(n, ca, c, cb)
                    
                    # Standard convention: 
                    # L-amino acids usually have a positive improper dihedral (~32 degrees)
                    # D-amino acids usually have a negative improper dihedral (~ -32 degrees)
                    chirality = "L" if angle > 0 else "D"
                    
                    results.append({
                        "resname": residue.get_resname(),
                        "resid": residue.id[1],
                        "chain": chain.id,
                        "angle": angle,
                        "chirality": chirality
                    })
                except KeyError:
                    # Missing one of the required atoms (N, CA, C, or CB)
                    continue

    return results