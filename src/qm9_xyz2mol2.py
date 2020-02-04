# Convert the .xyz files in QM9 dataset to .mol2 files

import os
# import multiprocessing as mp

import openbabel
from tqdm import tqdm


if __name__ == "__main__":
    files = os.scandir("../data/QM9/quantum-machine-9-aka-qm9")
    out_path = "../data/QM9/mol2"
    os.makedirs(out_path, exist_ok=True)
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("xyz", "mol2")
    for f in tqdm(list(files)):
        if f.is_dir():
            continue
        path, filename = os.path.split(f.path)
        output_file = os.path.join(
            os.path.dirname(path), "mol2", filename.split(".")[0]+".mol2")
        conv.OpenInAndOutFiles(f.path, output_file)
        conv.Convert()
        conv.CloseOutFile()
