import os
from collections import namedtuple
import tarfile

import numpy as np
import pandas as pd

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
        tar.list(verbose=False)



if __name__ == "__main__":
    t = TOUGH_C1()
    print(f"control: {t.control_ls[0:2]}")
    print(f"heme: {t.heme_ls[0:2]}")
    print(f"nucleotide: {t.nucleotide_ls[0:2]}")
    print(f"steroid: {t.steroid_ls[0:2]}")
    # t._load_tar("protein-control.tar.gz")
