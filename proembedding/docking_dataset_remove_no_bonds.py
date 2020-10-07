import os
import shutil
from utils.mol2parser import Mol2Parser
from argparse import ArgumentParser

def remove_no_bonds(path):
    mol2_files = os.scandir(path)
    trash_bin = os.path.join(path, "no_bonds")
    os.makedirs(trash_bin, exist_ok=True)
    for mf in mol2_files:
        if not mf.name.endswith(".mol2"):
            continue
        if not "pose1" in mf.name:
            continue
        m2p  = Mol2Parser(mf.path)
        if m2p.n_bonds == 0:
            shutil.move(mf.path, trash_bin)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--path")
    args = parser.parse_args()

    remove_no_bonds(args.path)
    