import os
from collections import namedtuple
import tarfile

import numpy as np
import pandas as pd

class TOUGH_C1:

    def __init__(self):
        # root path of the TOUGH-C1 data
        self.root = os.path.join(os.path.pardir, "data", "osfstorage-archive")
        # load the pdb id lists of the subsets
        self.control_ls = self._load_ls("deepdrug3d_control.lst")
        self.heme_ls = self._load_ls("deepdrug3d_heme.lst")
        self.nucleotide_ls = self._load_ls("deepdrug3d_nucleotide.lst")
        self.steroid_ls = self._load_ls("deepdrug3d_steroid.lst")

    def _load_ls(self, path):
        file_path = os.path.join(self.root, path)
        Protein = namedtuple("protein", "label pdbid")
        ret_ls = list()
        with open(file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            ret_ls.append(Protein(label=int(line[0]), pdbid=line[1:]))
        return ret_ls


if __name__ == "__main__":
    t = TOUGH_C1()
    print(t.heme_ls)
