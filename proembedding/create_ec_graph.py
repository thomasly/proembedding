import os
from random import sample
import pickle as pk

import pandas as pd

from utils.graph_creator import GraphCreator


def create_file_list(path="../data/Enzyme/EC_PDB"):
    entries = os.scandir(path)
    file_list = list()
    for e in entries:
        if e.is_file and e.name.endswith(".pdb"):
            file_list.append(e.path)
        elif e.is_dir:
            sub_list = create_file_list(e.path)
            file_list += sub_list
        else:
            continue
    return file_list


def create_labels(csv="../data/Enzyme/Structure_EC_benchmark.csv"):
    ec_df = pd.read_csv(csv)
    labels = dict()
    for _, row in ec_df.iterrows():
        if int(row["Chain Length"]) < 30:
            continue
        if str(row["EC No"]) != "nan":
            label = 1
        else:
            label = 0
        pdb_id = row["PDB ID"].upper()
        labels[pdb_id] = label
    return labels


def create_graph(cutoff, save_path, prefix, n_samples):
    pdb_file_list = create_file_list()
    pdb_file_list = sample(pdb_file_list, n_samples)
    with open("../data/Enzyme/EC_graph/sampled_file_list", "wb") as f:
        pk.dump(pdb_file_list, f)
    graph_label_dict = create_labels()
    graph_creator = GraphCreator(pdb_file_list, graph_label_dict)
    graph_creator.save_graph(prefix, cutoff, save_path)


if __name__ == "__main__":
    import sys
    log_f = open("log", "w")
    sys.stdout = log_f
    create_graph(10, "../data/Enzyme/EC_graph", "ec", 5000)
    sys.stdout.close()
