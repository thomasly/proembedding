import os
from utils.xyzparser import XYZParser
from tqdm import tqdm

files = os.scandir("../data/QM9/quantum-machine-9-aka-qm9")
atom_type = 0
atom_dict = dict()
for f in tqdm(files):
    if f.is_dir():
        continue
    parser = XYZParser(f.path)
    atoms = parser.atoms
    for a in atoms:
        if a in atom_dict:
            continue
        atom_dict[a] = atom_type
        atom_type += 1
print(atom_dict)
