import os
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np

from utils.mol2parser import Mol2toGraph


def _convert2string(array, precision=6):
    """ Convert a 2D numpy array to a writable string withou brackets
    """
    ret = ""
    for row in range(array.shape[0]):
        ret += ",".join(map(str, array[row].round(precision).tolist()))+"\n"
    return ret


def _get_graph_label(path):
    """ Get the graph label based on the path
    """
    if os.path.basename(os.path.dirname(path)) == "actives":
        return 1
    return 0


def save_graph(in_path, out_path, prefix, decimal=6):
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

    # open output files
    a = open(os.path.join(out_path, prefix+"_A.txt"), "w")
    idc = open(
        os.path.join(out_path, prefix+"_graph_indicator.txt"), "w")
    g_label = open(
        os.path.join(out_path, prefix+"_graph_labels.txt"), "w")
    n_label = open(
        os.path.join(out_path, prefix+"_node_labels.txt"), "w")
    edge_attr = open(
        os.path.join(out_path, prefix+"_edge_attributes.txt"), "w")
    mol_list = open(
        os.path.join(out_path, prefix+"_mol_list.txt"), "w")

    # initialize variables for graph indicator and nodes indices
    graph_id = 1
    node_starting_index = 0
    for infile in tqdm(infiles):
        if not infile.name.endswith(".mol2"):
            continue
        if "pose1" not in infile.name:
            continue
        m2g = Mol2toGraph(infile.path)

        # generate adjacency matrix
        adj_matrix = m2g.get_adjacency_matrix()
        # write bond types
        try:
            bond_types = m2g.get_bond_types()
        except KeyError:  # files lacking bond information
            continue
        for origin, target in zip(adj_matrix.row, adj_matrix.col):
            key = str(origin+1) + "-" + str(target+1)
            edge_attr.write(str(bond_types[key])+"\n")
        # write adjacency matrix and update node starting index
        for origin, target in zip(adj_matrix.row, adj_matrix.col):
            origin = origin + 1 + node_starting_index
            target = target + 1 + node_starting_index
            a.write(str(origin)+","+str(target)+"\n")
        node_starting_index += m2g.n_atoms

        # write graph indicator
        idc.write((str(graph_id)+"\n")*m2g.n_atoms)
        graph_id += 1

        # write graph label
        graph_label = _get_graph_label(infile.path)
        g_label.write(str(graph_label)+"\n")

        # write node features
        coordinates = np.array(m2g.atom_coordinates)
        surf_norms = m2g.get_surf_norms()
        atom_types = np.array(m2g.get_atom_types()).reshape(-1, 1)
        node_features = np.concatenate((coordinates, surf_norms, atom_types),
                                       axis=1)
        writable = _convert2string(node_features, decimal)
        n_label.write(writable)

        # log the mol2 file path
        mol_list.write(infile.path+"\n")

    # close output files
    a.close()
    idc.close()
    g_label.close()
    n_label.close()
    edge_attr.close()
    mol_list.close()


class MainParser(ArgumentParser):

    def __init__(self):
        super(MainParser, self).__init__()
        self.add_argument("-i", "--in-path",
                          help="Path to the dataset directories.")
        self.add_argument("-o", "--out-path",
                          help="Path to save the graph files.")
        self.add_argument("-p", "--prefix", default=None,
                          help="Prefix for the output files")
        self.add_argument("-d", "--decimal", type=int, default=6,
                          help="Precision digits of the floats.")


if __name__ == "__main__":
    parser = MainParser()
    args = parser.parse_args()
    in_path = args.in_path
    out_path = args.out_path
    prefix = args.prefix
    decimal = args.decimal

    save_graph(in_path, out_path, prefix, decimal)
