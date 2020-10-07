import os
import multiprocessing as mp
from urllib.request import urlopen

import pandas as pd


def retrieve_pdb(pdb_id):
    file_name = pdb_id + ".pdb"
    file_name = file_name.lower()
    des_folder = "../data/Enzyme/EC_PDB"
    try:
        print(f"retrieving {pdb_id}...", end="\r")
        request = urlopen(
            f"https://files.rcsb.org/download/{file_name}", timeout=2000)
        with open(f"{des_folder}/{file_name}", "wb") as f:
            f.write(request.read())     
        print(f"{pdb_id} retrieved...          ", end="\r")
    except Exception:
        return


if __name__ == "__main__":
    csv = "../data/Enzyme/Structure_EC_benchmark.csv"
    pdb_df = pd.read_csv(csv)
    pdb_df = pdb_df.loc[pdb_df["Chain Length"] > 30]
    n_pdbs = pdb_df.shape[0]

    des_folder = "../data/Enzyme/EC_PDB"
    os.makedirs(des_folder, exist_ok=True)

    pool = mp.Pool(12)
    pool.map_async(retrieve_pdb, pdb_df["PDB ID"])
    pool.close()
    pool.join()