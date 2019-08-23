import os
from collections import namedtuple
import tarfile

import numpy as np
import pandas as pd
from tqdm import tqdm

class TOUGH_C1:

    def __init__(self):
        # root path of the TOUGH-C1 data
        self.root = os.path.join(os.path.pardir, "data", "osfstorage-archive")
        # dict for label generation
        self.label_dict = {
            "control": 0,
            "heme": 1,
            "nucleotide": 2,
            "steroid": 3
        }
        # load the pdb id lists of the subsets
        self.control_ls = self._load_lst("deepdrug3d_control.lst")
        self.heme_ls = self._load_lst("deepdrug3d_heme.lst")
        self.nucleotide_ls = self._load_lst("deepdrug3d_nucleotide.lst")
        self.steroid_ls = self._load_lst("deepdrug3d_steroid.lst")

    def _load_lst(self, path):
        file_path = os.path.join(self.root, path)
        lb = self.label_dict[path.split(".")[0].split("_")[1]]
        Protein = namedtuple("protein", "label id")
        ret_ls = list()
        with open(file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            ret_ls.append(Protein(label=lb, id=line))
        return ret_ls

    def _load_tar(self, path):
        tar = tarfile.open(os.path.join(self.root, path))
        pdb_info = dict()
        Atom = namedtuple(
                    "atom",
                    "atom_id atom_name residue_id residue_name x y z"
                )
        for pdb in tqdm(tar.getmembers()):
            if not pdb.isfile():
                continue
            if pdb.name.split("/")[1].startswith("."):
                continue
            name = pdb.name.split("/")[1].split(".")[0]
            try:
                assert(len(name)==5)
            except AssertionError:
                print(f"{pdb.name} is not a valid pdb file")
                continue
            atom_ls = list()
            pdb_f = tar.extractfile(pdb)
            lines = pdb_f.readlines()
            pdb_f.close()
            for line in lines:
                if not line.startswith(b"ATOM"):
                    continue
                line = line.decode("utf-8")
                atom_id = int(line[6:11])
                atom_name = line[12:16].strip()
                residue_id = int(line[22:26])
                residue_name = line[17:20].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom = Atom(
                    atom_id, atom_name, residue_id, residue_name, x, y, z)
                atom_ls.append(atom)
            pdb_info[name] = atom_ls
        
        return pdb_info


if __name__ == "__main__":
    def print_first(d):
        key = list(d.keys())[0]
        print(key)
        print(d[key][0:10])

    t = TOUGH_C1()
    print(f"control: {t.control_ls[0:2]}")
    print(f"heme: {t.heme_ls[0:2]}")
    print(f"nucleotide: {t.nucleotide_ls[0:2]}")
    print(f"steroid: {t.steroid_ls[0:2]}")
    print_first(t._load_tar("protein-control.tar.gz"))
