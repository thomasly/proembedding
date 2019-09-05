import os
import sys
import tarfile
import pickle as pk
from random import sample, randrange
from collections import namedtuple
import argparse

import numpy as np
import freesasa as fs
from tqdm import tqdm

class PDBtoPFH():

    def __init__(self, file_path, cutoff=7):
        self.file_path = file_path
        self.cutoff = cutoff
        self.resi_to_int = {
            "ALA": 0,
            "CYS": 1, 
            "ASP": 2,
            "GLU": 3,
            "PHE": 4,
            "GLY": 5,
            "HIS": 6,
            "ILE": 7, 
            "LYS": 8,
            "LEU": 9,
            "MET": 10,
            "ASN": 11,
            "PRO": 12,
            "GLN": 13,
            "ARG": 14,
            "SER": 15,
            "THR": 16,
            "VAL": 17,
            "TRP": 18,
            "TYR": 19,
            "NAA": 20 # not a amino acid
        }
        self.Atom = namedtuple(
            "atom",
            "atom_id atom_name residue_id residue_name x y z")

    def get_attributes(self):
        # read pdb file
        with open(self.file_path, "r") as f:
            self.data = f.readlines()
        # calculate solvent access data
        self.solvent_access = fs.calc(fs.Structure(self.file_path))
        self._clean_data()
        self._ca_attributes()
        self._distance_to_others()
        self._find_in_range()

    # clean data, save as {atom_type/residue_index: coordinates}
    def _clean_data(self):
        self.cl_data = list()
        for d in self.data:
            if not d.startswith("ATOM"):
                continue
            try:
                atom_id = int(d[6:11].strip())
                res_id = int(d[22:26].strip())
                atom_name = d[12:16].strip()
                res_name = d[17:20].strip()
                x = float(d[30:38])
                y = float(d[38:46])
                z = float(d[46:54])
                atom = self.Atom(
                    atom_id=atom_id, atom_name=atom_name, residue_id=res_id,
                    residue_name=res_name, x=x, y=y, z=z
                )
                self.cl_data.append(atom)
            except ValueError:
                continue

    # get the coordinates of all C-alpha
    def _get_ca(self):
        self.all_ca = []
        self.ca_res = []
        self.ca_solv = []
        for atom in self.cl_data:
            if atom.atom_name.upper() == "CA":
                try:
                    self.ca_res.append(
                        self.resi_to_int[atom.residue_name.upper()])
                except KeyError:
                    self.ca_res.append(self.resi_to_int["NAA"])
                try:
                    self.ca_solv.append(
                        self.solvent_access.atomArea(atom.atom_id))
                except Exception:
                    self.ca_solv.append(0.0)
                self.all_ca.append([atom.x, atom.y, atom.z])

    def _get_c(self):
        self.all_c = []
        for atom in self.cl_data:
            if atom.atom_name.upper() == "C":
                self.all_c.append([atom.x, atom.y, atom.z])

    # ### calculate the distance of all the C-alpha to other C-alpha
    def distance(self, start, end):
        st = np.array(start)
        ed = np.array(end)
        return np.linalg.norm(st-ed)

    def _distance_to_others(self):
        self.dist_to_others = []
        for ca_st in (self.all_ca):
            dist_st_ed = []
            for ca_ed in (self.all_ca):
                dist_st_ed.append(self.distance(ca_st, ca_ed))
            self.dist_to_others.append(dist_st_ed)

    # ### Find the C-alpha within the range of CUTOFF
    def _find_in_range(self):
        self.within_range = []
        for i, ds in enumerate(self.dist_to_others):
            neighbors = []
            for j, d in enumerate(ds):
                if d < self.cutoff and d > 0:
                    key = str(i) + "/" + str(j)
                    neighbors.append(key)
            self.within_range.append(neighbors)

    # ### calculate the "surface norms" of all the C-alpha
    def _calculate_norms(self):
        assert(len(self.all_ca)==len(self.all_c))
        all_ca = np.array(self.all_ca, dtype=np.float32)
        all_c = np.array(self.all_c, dtype=np.float32)
        self.norms = all_c - all_ca

    def _ca_attributes(self):
        self._get_ca()
        self._get_c()
        self._calculate_norms()
        self.ca_attributes = [
            ca + list(sf) + [resi_type] + [solv] for \
                ca, sf, resi_type, solv in zip(
                    self.all_ca, self.norms, self.ca_res, self.ca_solv)]


class Pocket2PFH(PDBtoPFH):

    def __init__(self, file_path, residue_list, cutoff=7):
        super(Pocket2PFH, self).__init__(file_path, cutoff)
        self.residue_list = residue_list

    # get the coordinates of all C-alpha in the residue list
    def _get_ca(self):
        self.all_ca = []
        self.ca_res = []
        self.ca_solv = []
        for atom in self.cl_data:
            if atom.atom_name.upper() == "CA" and (
                    atom.residue_id in self.residue_list):
                try:
                    self.ca_res.append(
                        self.resi_to_int[atom.residue_name.upper()])
                except KeyError:
                    self.ca_res.append(self.resi_to_int["NAA"])
                try:
                    self.ca_solv.append(
                        self.solvent_access.atomArea(atom.atom_id))
                except Exception:
                    self.ca_solv.append(0.0)
                self.all_ca.append([atom.x, atom.y, atom.z])

    def _get_c(self):
        self.all_c = []
        for atom in self.cl_data:
            if atom.atom_name.upper() == "C" and (
                    atom.residue_id in self.residue_list):
                self.all_c.append([atom.x, atom.y, atom.z]) 

    

                
class TOUGH_C1_Graph:

    def __init__(self, dataset):
        self.root = os.path.join(os.path.pardir, "data")
        self.tough_dir = os.path.join(self.root, "osfstorage-archive")
        self.control = os.path.join(self.tough_dir, "protein-control")
        self.heme = os.path.join(self.tough_dir, "protein-heme")
        self.nucleotide = os.path.join(self.tough_dir, "protein-nucleotide")
        self.steroid = os.path.join(self.tough_dir, "protein-steroid")
        try:
            assert(dataset in ["tough-c1", "nucleotide", "heme"])
            self.dataset = dataset
            if dataset == "tough-c1":
                self.data_list = [
                    self.control, self.heme, self.nucleotide, self.steroid]
            elif dataset == "nucleotide":
                self.data_list = [self.control, self.nucleotide]
            elif dataset == "heme":
                self.data_list = [self.control, self.heme]
        except AssertionError:
            raise
        self.save_path = os.path.join(self.root, "graphs")
        os.makedirs(self.save_path, exist_ok=True)

    def create_file_list(self, dir_path):
        """ Create file list of the dataset
        """
        file_list = list()
        for f in os.scandir(dir_path):
            if f.name.endswith(".pdb"):
                file_list.append(f)
        return file_list

    def get_adjacency_matrix(self, pfh_class, starting_number):
        """ Save data into graph format
        """
        edges = []
        for d in pfh_class.within_range:
            for k in d:
                edges.append(list(map(int, k.split("/"))))
        ret = ""
        for e in edges:
            i = 0
            while i < len(e):
                e[i] += starting_number
                i += 1
            ret += str(e)[1:-1] + "\n"
        return ret
    
    def save_graph(self,
                   file_prefix: str,
                   cutoff=10):
        """Convert pdb files to graph
        """
        save_path = os.path.join(self.save_path, self.dataset, str(cutoff))
        os.makedirs(save_path, exist_ok=True)

        a = open(os.path.join(save_path, file_prefix+"_A.txt"), "w")
        idc = open(
            os.path.join(save_path, file_prefix+"_graph_indicator.txt"), "w")
        g_label = open(
            os.path.join(save_path, file_prefix+"_graph_labels.txt"), "w")
        n_label = open(
            os.path.join(save_path, file_prefix+"_node_labels.txt"), "w")

        starting_number = 1
        pdb_id = 1
        class_label = 0
        for subset in self.data_list:
            file_list = self.create_file_list(subset)
            for pdb in tqdm(file_list):
                if not pdb.name.endswith(".pdb"):
                    continue
                pdb_pfh = PDBtoPFH(pdb.path, cutoff=cutoff)
                pdb_pfh.get_attributes()
                a.write(self.get_adjacency_matrix(pdb_pfh, starting_number))
                idc.write((str(pdb_id)+"\n")*len(pdb_pfh.all_ca))
                g_label.write(str(class_label)+"\n")
                for attr in pdb_pfh.ca_attributes:
                    n_label.write(str(attr)[1:-1]+"\n")
                starting_number += len(pdb_pfh.all_ca)
                pdb_id += 1
            class_label += 1

        a.close()
        idc.close()
        g_label.close()
        n_label.close()


class TOUGH_C1_Pocket_Graph(TOUGH_C1_Graph):

    def __init__(self, dataset, pocket_type="lpc"):
        super(TOUGH_C1_Pocket_Graph, self).__init__(dataset)
        self.pocket_type = pocket_type
        self.resi_dict = self._create_resi_dict()

    def _create_resi_dict(self):
        file_list = [self.pocket_type+"-"+suffix+".residues" for suffix in [
            "control", "nucleotide", "heme", "steroid"]]
        resi_dict = dict()
        for resi_f in file_list:
            path = os.path.join(self.tough_dir, resi_f)
            with open(path, "r") as f:
                for line in f.readlines():
                    if len(line) < 5:
                        continue
                    pdb_id = line[:5]
                    resi_ids = list()
                    for token in line.split()[4:]:
                        resi_ids.append(int(token[1:]))
                    resi_dict[pdb_id] = resi_ids
        return resi_dict

    def save_graph(self,
                   file_prefix: str,
                   cutoff=10):
        """Convert pdb files to graph
        """
        save_path = os.path.join(
            self.save_path, self.dataset+"_pocket", str(cutoff))
        os.makedirs(save_path, exist_ok=True)

        a = open(os.path.join(save_path, file_prefix+"_A.txt"), "w")
        idc = open(
            os.path.join(save_path, file_prefix+"_graph_indicator.txt"), "w")
        g_label = open(
            os.path.join(save_path, file_prefix+"_graph_labels.txt"), "w")
        n_label = open(
            os.path.join(save_path, file_prefix+"_node_labels.txt"), "w")

        starting_number = 1
        pdb_id = 1
        class_label = 0
        for subset in self.data_list:
            file_list = self.create_file_list(subset)
            for pdb in tqdm(file_list):
                if not pdb.name.endswith(".pdb"):
                    continue
                residue_list = self.resi_dict[pdb.name.split(".")[0]]
                pdb_pfh = Pocket2PFH(pdb.path, residue_list, cutoff=cutoff)
                pdb_pfh.get_attributes()
                a.write(self.get_adjacency_matrix(pdb_pfh, starting_number))
                idc.write((str(pdb_id)+"\n")*len(pdb_pfh.all_ca))
                g_label.write(str(class_label)+"\n")
                for attr in pdb_pfh.ca_attributes:
                    n_label.write(str(attr)[1:-1]+"\n")
                starting_number += len(pdb_pfh.all_ca)
                pdb_id += 1
            class_label += 1

        a.close()
        idc.close()
        g_label.close()
        n_label.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pocket", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.pocket:
        tcpg = TOUGH_C1_Pocket_Graph("tough-c1")
        tcpg.save_graph("TOUGH-C1")
        tcpg.save_graph("TOUGH-C1", 15)
        tcpg = TOUGH_C1_Pocket_Graph("nucleotide")
        tcpg.save_graph("nucleotide")
        tcpg.save_graph("nucleotide", 15)
        tcpg = TOUGH_C1_Pocket_Graph("heme")
        tcpg.save_graph("heme")
        tcpg.save_graph("heme", 15)
    else:
        tcg = TOUGH_C1_Graph("tough-c1")
        tcg.save_graph("TOUGH-C1")
        tcg.save_graph("TOUGH-C1", 15)
        tcg.save_graph("TOUGH-C1", 20)
        tcg = TOUGH_C1_Graph("nucleotide")
        tcg.save_graph("nucleotide")
        tcg.save_graph("nucleotide", 15)
        tcg.save_graph("nucleotide", 20)
        tcg = TOUGH_C1_Graph("heme")
        tcg.save_graph("heme")
        tcg.save_graph("heme", 15)
        tcg.save_graph("heme", 20)
