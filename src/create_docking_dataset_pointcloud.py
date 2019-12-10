import os
import pickle as pk
from argparse import ArgumentParser

from utils.mol2parser import Mol2toPointCloud
from tqdm import tqdm

def save_point_cloud(in_path,
              out_path,
              include_type=True,
              prefix=None):
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

    # generate all the point clouds
    clouds = dict()
    for mol_f in tqdm(infiles):
        if not mol_f.name.endswith(".mol2"):
            continue
        if "pose1" not in mol_f.name:
            continue
        mol_id = mol_f.name.split(".")[0]
        mol = Mol2toPointCloud(mol_f.path)
        point_cloud = mol.get_point_cloud(include_type=include_type)
        clouds[mol_id] = point_cloud
    with open(os.path.join(out_path, prefix+".pointcloud"), "wb") as f:
        pk.dump(clouds, f)


class MainParser(ArgumentParser):

    def __init__(self):
        super(MainParser, self).__init__()
        self.add_argument("-i", "--in-path",
                          help="Path to the dataset directories.")
        self.add_argument("-o", "--out-path",
                          help="Path to save the graph files.")
        self.add_argument("-p", "--prefix", default=None,
                          help="Prefix for the output files.")
        self.add_argument("-t", "--type-channel", action="store_true",
                          help="Include type channel in the point clouds.")


if __name__ == "__main__":
    parser = MainParser()
    args = parser.parse_args()
    in_path = args.in_path
    out_path = args.out_path
    prefix = args.prefix
    include_type = args.type_channel

    save_point_cloud(in_path, out_path, include_type, prefix)
