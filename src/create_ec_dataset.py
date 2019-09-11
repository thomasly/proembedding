import os
import random
import pickle as pk

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
tqdm.pandas()
# import markov_clustering as mc

def get_pdb_id(name):
    if name.startswith("d"):
        return name[1:5].upper()
    else:
        return name[4:8].upper()

def calculate_weight(data):
    rmsd = float(data[0])
    cov1 = float(data[1])
    cov2 = float(data[2])
    return min(1.0, 0.01 / (rmsd + 1e-5) * (cov1 + cov2) / 2)

def get_pdb_id_map(row, col):
    try:
        with open("../data/Enzyme/pdb_id_map", "rb") as f:
            return pk.load(f)
    except FileNotFoundError:
        id_map = dict()
        counter = 0
        _, uniques = pd.factorize(row.append(col))
        for pdb_id in uniques:
            if pdb_id in id_map:
                continue
            else:
                id_map[pdb_id] = counter
                counter += 1
        with open("../data/Enzyme/pdb_id_map", "wb") as f:
            pk.dump(id_map, f)
        return id_map


def read_csv(nrows=None):
    print("Loading csv file...")
    df = pd.read_csv('../data/Enzyme/fatcat_rigid_pdb_all.txt.gz',
                     compression='gzip',
                     header=11,
                     sep='\t',
                     quotechar='"',
                     error_bad_lines=False,
                     nrows=nrows)
    print("Done.")
    return df


def analyze_df(df, id_to_int=True):
    print("Converting pdb ids to integers...(1/2)")
    row_ind = df["#name1"].progress_map(get_pdb_id)
    print("Converting pdb ids to integers...(2/2)")
    col_ind = df["name2"].progress_map(get_pdb_id)
    if id_to_int:
        pdb_id_map = get_pdb_id_map(row_ind, col_ind)
        row_ind = row_ind.map(pdb_id_map)
        col_ind = col_ind.map(pdb_id_map)
    print("Done.")
    print("Calculating weights...")
    data = df[["rmsd", "cov1", "cov2"]].progress_apply(
        calculate_weight, axis=1)
    print("Done.")
    return row_ind, col_ind, data


def create_csr_matrix(test=False):
    if test:
        df = read_csv(100000)
    else:
        df = read_csv()
    row_ind, col_ind, data = analyze_df(df)
    
    m = max(row_ind.max()+1, col_ind.max()+1)
    matrix = csr_matrix((data, (row_ind, col_ind)),
                        shape=(m, m))
    with open("../data/Enzyme/matrix", "wb") as f:
        pk.dump(matrix, f)
    return matrix


def write_abc_file(test=False):
    abc_f = open("../data/Enzyme/fatcat.abc", "w")
    if test:
        df = read_csv(100000)
    else:
        df = read_csv()
    row_ind, col_ind, data = analyze_df(df, id_to_int=False)
    print("Writing matrix to .abc file...")
    for r, c, d in tqdm(zip(row_ind, col_ind, data), total=len(data)):
        abc_f.write(r+"\t"+c+"\t"+str(d)+"\n")
    print("Done.")
    abc_f.close()


def create_cluster_dict(cluster_f):
    with open(cluster_f, "r") as f:
        clusters = f.readlines()
    cluster_dict = dict()
    idx = 0
    for line in clusters:
        if len(line) < 4:
            continue
        pdbs = line.strip().split()
        for pdb in pdbs:
            cluster_dict[pdb] = idx
        idx += 1
    return cluster_dict


def ec_cluster_from_csv(cluster_dict):
    ec_df = pd.read_csv("../data/Enzyme/Structure_EC_benchmark.csv")
    ec_clusters = list()
    cluster_ids = list()
    labels = dict()
    extra_cluster = 0
    for _, row in ec_df.iterrows():
        if int(row["Chain Length"]) < 30:
            continue
        if str(row["EC No"]) != "nan":
            label = 1
        else:
            label = 0
        pdb_id = row["PDB ID"]
        labels[pdb_id] = label
        try:
            cluster_id = cluster_dict[pdb_id]
        except KeyError:
            cluster_id = len(cluster_dict) + extra_cluster
            extra_cluster += 1
        if cluster_id not in cluster_ids:
            cluster_ids.append(cluster_id)
            ec_clusters.append(set([pdb_id]))
        else:
            ec_idx = cluster_ids.index(cluster_id)
            ec_clusters[ec_idx].add(pdb_id)
    return ec_clusters, labels



if __name__ == "__main__":
    # try:
    #     with open("../data/Enzyme/matrix", "rb") as f:
    #         matrix = pk.load(f)
    # except FileNotFoundError:
    #     matrix = create_csr_matrix()
    # # print(matrix[matrix >= 0.1].shape)
    # # print(matrix.shape)
    # # dok = mc.dok_matrix(matrix.shape)
    # # print(dok[matrix[matrix >= 0.1]].shape)
    # # print(dok.shape)
    # result = mc.run_mcl(matrix,
    #                     inflation=2,
    #                     pruning_threshold=0,
    #                     verbose=1)
    # clusters = mc.get_clusters(result)
    # print(len(clusters))
    # print(matrix.shape)
    # with open("../data/Enzyme/clusters", "wb") as f:
    #     pk.dump(clusters, f)
    write_abc_file()