import os
import pickle as pk
from argparse import ArgumentParser

from utils.mol2parser import Mol2toGrid
from tqdm import tqdm

def save_grid(in_path,
              out_path,
              grid_dimension,
              zoom,
              prefix=None,
              rotate=False):
    # define prefix
    if prefix is None:
        prefix = os.path.basename(in_path)

    # paths to the active and decoy directories
    active_path = os.path.join(in_path, "actives")
    inactive_path = os.path.join(in_path, "decoys")
    # input file list
    infiles = list(os.scandir(active_path)) + list(os.scandir(inactive_path))
    # create output directory
    os.makedirs(out_path, exist_ok=True)

    # generate all the grids
    grids = dict()
    for mol_f in tqdm(infiles):
        if not mol_f.name.endswith(".mol2"):
            continue
        if "pose1" not in mol_f.name:
            continue
        mol_id = mol_f.name.split(".")[0]
        mol = Mol2toGrid(mol_f.path)
        mol_grid = mol.get_grid(grid_dimension, zoom=zoom, rotate=rotate)
        grids[mol_id] = mol_grid
        break
    if rotate:
        extention = ".pca.grids"
    else:
        extention = ".grids"
    with open(os.path.join(out_path, prefix+extention), "wb") as f:
        pk.dump(grids, f)


class MainParser(ArgumentParser):

    def __init__(self):
        super(MainParser, self).__init__()
        self.add_argument("-i", "--in-path",
                          help="Path to the dataset directories.")
        self.add_argument("-o", "--out-path",
                          help="Path to save the graph files.")
        self.add_argument("-p", "--prefix", default=None,
                          help="Prefix for the output files.")
        self.add_argument("-d", "--dimension", type=int, default=33,
                          help="Dimension of the grids.")
        self.add_argument("-z", "--zoom", type=float, default=1.0,
                          help="Fold to zoom in when saving the molecule.")
        self.add_argument("-r", "--rotate", action="store_true",
                          help="Whether to align the coordinates to PCA "
                               "componensts.")


if __name__ == "__main__":
    parser = MainParser()
    args = parser.parse_args()
    in_path = args.in_path
    out_path = args.out_path
    prefix = args.prefix
    dimension = args.dimension
    zoom = args.zoom
    rotate = args.rotate

    save_grid(in_path, out_path, dimension, zoom, prefix, rotate)

