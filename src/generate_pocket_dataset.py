def create_residue_dict(residues_f):
    with open(residues_f, "r") as f:
        lines = f.readlines()

    residue_dict = dict()
    for line in lines:
        if len(line) < 5:
            continue
        tokens = line.strip().split()
        pdb_id = tokens[0][:5]
        residues = [int(resi[1:]) for resi in tokens[4:]]
        residue_dict[pdb_id] = residues
    return residue_dict


if __name__ == "__main__":
    import pickle as pk
    import os
    import sys
    import argparse

    from utils.pdb2grid import PDB2Grid
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--set", type=str, required=True)
    args = parser.parse_args(sys.argv[1:])
    data_path = os.path.join(os.path.pardir, "data", "tough_c1")
    pdb_folder = os.path.join(data_path, "protein-"+args.set)
    residue_file = os.path.join(data_path, "lpc-"+args.set+".residues")
    residue_dic = create_residue_dict(residue_file)
    pdb_files = os.scandir(pdb_folder)
    grids = dict()
    grids_resi = dict()
    for pdb_f in tqdm(list(pdb_files)):
        if not pdb_f.name.endswith(".pdb"):
            continue
        pdb_id = pdb_f.name.split(".")[0]
        pdb = PDB2Grid(pdb_f.path)
        pocket_grid = pdb.get_pocket_ca_grid(residue_dic[pdb_id])
        pocket_resi_grid = pdb.get_pocket_ca_res_grid(residue_dic[pdb_id])
        grids[pdb_id] = pocket_grid
        grids_resi[pdb_id] = pocket_resi_grid
    with open(os.path.join(data_path, args.set+"-pocket.grids"), "wb") as f:
        pk.dump(grids, f)
    with open(os.path.join(data_path, args.set+"-pocket.resigrids"),
              "wb") as f:
        pk.dump(grids_resi, f)
