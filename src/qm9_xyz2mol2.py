# Convert the .xyz files in QM9 dataset to .mol2 files

import os
import multiprocessing as mp

import openbabel


def xyz2mol(file_paths):
    for file_path in file_paths:
        if os.path.isdir(file_path):
            continue
        path, filename = os.path.split(file_path)
        conv = openbabel.OBConversion()
        conv.SetInAndOutFormats("xyz", "mol2")
        output_file = os.path.join(
            os.path.dirname(path), "mol2", filename.split(".")[0]+".mol2")
        conv.OpenInAndOutFiles(file_path, output_file)
        conv.Convert()
        conv.CloseOutFile()
        del conv


if __name__ == "__main__":
    files = os.scandir("../data/QM9/quantum-machine-9-aka-qm9")
    paths = [f.path for f in files]
    n_chunks = int(os.cpu_count()-2)
    chunks = [paths[i:i+n_chunks] for i in range(0, len(paths), n_chunks)]
    out_path = "../data/QM9/mol2"
    os.makedirs(out_path, exist_ok=True)
    procs = list()
    for c in chunks:
        proc = mp.Process(target=xyz2mol, args=(c,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
