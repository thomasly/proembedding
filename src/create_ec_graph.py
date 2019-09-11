import os

import pandas as pd


def create_file_list(path="../data/Enzyme/PDB"):
    entries = os.scandir(path)
    file_list = list()
    for e in entries:
        if e.is_file and (e.name.endswith(".pdb") or e.name.endswith(".cif")):
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


def create_graph(cutoff, save_path, prefix):
    pass


if __name__ == "__main__":
    pass