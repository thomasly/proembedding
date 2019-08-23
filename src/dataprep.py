import os
from collections import namedtuple
import tarfile
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

class TOUGH_C1:

    def __init__(self, random_seed=0, train_test_ratio=0.9):
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
        # create training and testing sets
        self.random_seed = random_seed
        self.train_ls, self.test_ls = self._split_dataset(train_test_ratio)
        
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
        print(f"loading {path} ...")
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

    def _split_dataset(self, train_test_ratio):
        random.seed(self.random_seed)
        random.shuffle(self.control_ls)
        random.shuffle(self.heme_ls)
        random.shuffle(self.nucleotide_ls)
        random.shuffle(self.steroid_ls)

        train_ls = list()
        test_ls = list()
        self._list_extend(train_ls, test_ls, self.control_ls, train_test_ratio)
        self._list_extend(train_ls, test_ls, self.heme_ls, train_test_ratio)
        self._list_extend(
            train_ls, test_ls, self.nucleotide_ls, train_test_ratio)
        self._list_extend(train_ls, test_ls, self.steroid_ls, train_test_ratio)
        random.shuffle(train_ls)
        random.shuffle(test_ls)
        return train_ls, test_ls

    def _list_extend(self, train_ls, test_ls, target_ls, ratio):
        splitor = int(len(target_ls)*ratio)
        train_ls += target_ls[:splitor]
        test_ls += target_ls[splitor:]

class TOUGH_POINT(TOUGH_C1):

    def __init__(self, random_seed=0, train_test_ratio=0.9, subset="train"):
        super(TOUGH_POINT, self).__init__(random_seed, train_test_ratio)
        self.subset = subset

    def __next__(self):
        pass



if __name__ == "__main__":
    def print_first(d):
        key = list(d.keys())[0]
        print(key)
        print(d[key][0:10])

    t = TOUGH_C1()
    # print(f"control: {t.control_ls[0:2]}")
    # print(f"heme: {t.heme_ls[0:2]}")
    # print(f"nucleotide: {t.nucleotide_ls[0:2]}")
    # print(f"steroid: {t.steroid_ls[0:2]}")
    # print_first(t._load_tar("protein-control.tar.gz"))
    print(len(t.train_ls))
    print(len(t.test_ls))
    print(t.train_ls[:10])
    print(t.test_ls[:10])
    print(len(t.control_ls))
    print(len(t.heme_ls))
    print(len(t.nucleotide_ls))
    print(len(t.steroid_ls))
