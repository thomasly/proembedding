import os
from argparse import ArgumentParser
import multiprocessing as mp
from functools import partial

from tqdm import tqdm

from utils.xyzparser import XYZ2Graph


def _convert2string(array, precision=6):
    """ Convert a 2D numpy array to a writable string without brackets
    """
    ret = ""
    for row in range(array.shape[0]):
        ret += ",".join(map(str, array[row].round(precision).tolist()))+"\n"
    return ret


def create_graph(infile, cutoff, precision):
    if not infile.endswith(".xyz"):
        return
    if os.path.isdir(infile):
        return
    m2g = XYZ2Graph(infile)
    # generate adjacency matrix
    adj_matrix = m2g.get_adjacency_matrix(cutoff=cutoff)
    # get graph indicator
    indicator_len = m2g.n_nodes
    # get graph label
    g_label = str(m2g.graph_label)+"\n"
    # get node features
    node_features = _convert2string(m2g.node_attributes, precision)
    # log the mol2 file path
    file_path = infile + "\n"

    return [adj_matrix, indicator_len, g_label, node_features, file_path]


def save_graph(in_path, out_path, prefix, cutoff=5, precision=6):
    # define prefix
    if prefix is None:
        prefix = os.path.basename(in_path)

    # input file list
    infiles = [f.path for f in os.scandir(in_path)]
    # create output directory
    os.makedirs(out_path, exist_ok=True)

    # initialize multiprocessing
    pool = mp.Pool(mp.cpu_count())
    create_graphs = partial(create_graph, cutoff=cutoff, precision=precision)

    # get graphs
    graphs = pool.map(create_graphs, list(infiles))

    # open output files
    a = open(os.path.join(out_path, prefix+"_A.txt"), "w")
    idc = open(
        os.path.join(out_path, prefix+"_graph_indicator.txt"), "w")
    g_label = open(
        os.path.join(out_path, prefix+"_graph_labels.txt"), "w")
    n_label = open(
        os.path.join(out_path, prefix+"_node_labels.txt"), "w")
    mol_list = open(
        os.path.join(out_path, prefix+"_mol_list.txt"), "w")

    # initialize variables for graph indicator and nodes indices
    graph_id = 1
    node_starting_index = 0

    for grp in tqdm(graphs):
        if grp is None:
            continue

        # write adjacency matrix and update node starting index
        adj_matrix = grp[0]
        for origin, target in zip(adj_matrix.row, adj_matrix.col):
            origin = origin + 1 + node_starting_index
            target = target + 1 + node_starting_index
            a.write(str(origin)+","+str(target)+"\n")
        node_starting_index += grp[1]

        # write graph indicator
        idc.write((str(graph_id)+"\n")*grp[1])
        graph_id += 1

        # write graph label
        g_label.write(grp[2])

        # write node features
        n_label.write(grp[3])

        # log the mol2 file path
        mol_list.write(grp[4])

    # close output files
    a.close()
    idc.close()
    g_label.close()
    n_label.close()
    mol_list.close()


class MainParser(ArgumentParser):

    def __init__(self):
        super(MainParser, self).__init__()
        self.add_argument("-i", "--in-path",
                          help="Path to the dataset directories.")
        self.add_argument("-o", "--out-path",
                          help="Path to save the graph files.")
        self.add_argument("-p", "--prefix", default=None,
                          help="Prefix for the output files.")
        self.add_argument("-c", "--cutoff", type=int, default=5,
                          help="The cut off value for adjacency matrix.")
        self.add_argument("-d", "--precision", type=int, default=6,
                          help="Precision digits of the floats.")


if __name__ == "__main__":
    parser = MainParser()
    args = parser.parse_args()
    save_graph(**vars(args))
