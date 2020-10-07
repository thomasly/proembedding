import os
from collections import namedtuple


class PDB:

    def __init__(self, pdb_f: str):
        self.pdb_f = pdb_f
        self.pdb_contents = None
        self.Atom = namedtuple(
            "atom",
            ["atom_id", "atom_name", "resi_name", "chain", "resi_id",
             "x", "y", "z", "occupancy", "b_factor", "element"]
        )
        self.atoms = None
        self.ca = None
        self.c = None
        self.n = None

    def _load_contents(self):
        with open(self.pdb_f, "r") as f:
            self.pdb_contents = f.readlines()

    def get_atoms(self):
        if self.pdb_contents is None:
            self._load_contents()
        if self.atoms is not None:
            return self.atoms
        self.atoms = list()
        self.ca = list()
        self.c = list()
        self.n = list()
        for line in self.pdb_contents:
            if not line.startswith("ATOM"):
                continue
            atom = self.Atom(
                int(line[6:11]),
                line[12:16].strip().upper(),
                line[17:20].strip().upper(),
                line[21].upper(),
                int(line[22:26]),
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
                float(line[54:60]),
                float(line[60:66]),
                line[76:78].strip().upper()
            )
            if atom.atom_name == "CA":
                self.ca.append(atom)
            if atom.atom_name == "C":
                self.c.append(atom)
            if atom.atom_name == "N":
                self.n.append(atom)
            self.atoms.append(atom)
        return self.atoms

    def get_ca(self):
        if self.ca is None:
            self.get_atoms()
        return self.ca

    def get_c(self):
        if self.c is None:
            self.get_atoms()
        return self.c

    def get_n(self):
        if self.n is None:
            self.get_atoms()
        return self.n

    @staticmethod
    def resi2int(resi_name: str):
        resi_name = resi_name.upper()
        resi_dict = {
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
        try:
            resi_int = resi_dict[resi_name]
        except KeyError:
            resi_int = resi_dict["NAA"]
        return resi_int


if __name__ == "__main__":
    pdb = PDB("../../data/Enzyme/EC_PDB/1a0b.pdb")
    
    print(pdb.get_ca()[0])
    print(pdb.get_c()[0])
    print(pdb.get_atoms()[0])