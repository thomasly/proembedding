from collections import namedtuple

import numpy as np
from scipy.sparse import coo_matrix


class Mol2Parser:
    
    def __init__(self, mol2_path):
        self._path = mol2_path

    @property
    def contents(self):
        try:
            return self._contents
        except AttributeError:
            self._contents = self._read_file()
            return self._contents

    @property
    def content_dict(self):
        try:
            return self._c_dict
        except AttributeError:
            self._c_dict = self._contents2dict()
            return self._c_dict

    @property
    def n_atoms(self):
        try:
            return self._n_atoms
        except AttributeError:
            self._n_atoms = len(self.content_dict["ATOM"])
            return self._n_atoms

    @property
    def n_bonds(self):
        try:
            return self._n_bonds
        except AttributeError:
            self._n_bonds = len(self.content_dict["BOND"])
            return self._n_bonds

    def _read_file(self):
        with open(self._path, "r") as f:
            return f.read()

    def _contents2dict(self):
        contents = self.contents.strip().split("\n")
        contents_dict = dict()
        current_key = "<TEMP>"
        contents_dict[current_key] = list()
        for line in contents:
            if line.startswith("@"):
                try:
                    current_key = line.split("<TRIPOS>")[1]
                    contents_dict[current_key] = list()
                    continue
                except IndexError:
                    current_key = "<TEMP>"
                    continue
            contents_dict[current_key].append(line.strip())
        contents_dict.pop("<TEMP>")
        return contents_dict

    def get_atom_attributes(self):
        try:
            return self._atom_attributes
        except AttributeError:
            Atom = namedtuple("atom", "id name x y z type subst_id "
                                      "subst_name charge")
            self._atom_attributes = list()
            for atom in self.content_dict["ATOM"]:
                self._atom_attributes.append(Atom(*atom.split()))
            return self._atom_attributes

    def get_bond_attributes(self):
        try:
            return self._bond_attributes
        except AttributeError:
            Bond = namedtuple("bond", "id origin target type")
            self._bond_attributes = list()
            for bond in self.content_dict["BOND"]:
                self._bond_attributes.append(Bond(*bond.split()))
            return self._bond_attributes


class Mol2toGraph:

    _atom_types = "C.3,C.2,C.1,C.ar,C.cat,N.3,N.2,N.1,N.ar,N.am,N.pl3,N.4,"\
                 "O.3,O.2,O.co2,O.spc,O.t3p,S.3,S.2,S.O,S.O2,P.3,F,H,H.spc,"\
                 "H.t3p,LP,Du,Du.C,Hal,Het,Hev,Li,Na,Mg,Al,Si,K,Ca,Cr.thm,"\
                 "Cr.oh,Mn,Fe,Co.oh,Cu,Any".split(",")
    _atom2int = {atom.upper():idx for idx, atom in enumerate(_atom_types)}

    _bond_types = "1,2,3,am,ar,du,un,cn,any".split(",")
    _bond2int = {bond.upper():idx for idx, bond in enumerate(_bond_types)}

    def __init__(self, mol2_path):
        self._path = mol2_path
        self.mol_parser = Mol2Parser(mol2_path)

    def bonds(self):
        bonds = list()
        for bond in self.mol_parser.get_bond_attributes():
            bonds.append([int(bond.origin), int(bond.target)])
        return bonds

    @property
    def atom_coordinates(self):
        try:
            return self._atom_coordinates
        except AttributeError:
            self._atom_coordinates = list()
            for atom in self.mol_parser.get_atom_attributes():
                self._atom_coordinates.append(
                    list(map(float, [atom.x, atom.y, atom.z])))
            return self._atom_coordinates

    @property
    def n_atoms(self):
        return self.mol_parser.n_atoms

    @property
    def n_bonds(self):
        return self.mol_parser.n_bonds

    def get_adjacency_matrix(self, sparse=True):
        bonds = self.bonds()
        dense = np.zeros((self.n_atoms, self.n_atoms))
        for bond in bonds:
            dense[bond[0]-1, bond[1]-1] = 1
            dense[bond[1]-1, bond[0]-1] = 1
        if not sparse:
            return dense
        else:
            return coo_matrix(dense)

    def _find_center(self):
        coor = np.array(self.atom_coordinates)
        return np.mean(coor, axis=0)

    def get_surf_norms(self):
        mol_center = self._find_center()
        coor = np.array(self.atom_coordinates)
        return coor - mol_center

    def get_atom_types(self):
        atom_types = list()
        for atom in self.mol_parser.get_atom_attributes():
            try:
                atom_types.append(self._atom2int[atom.type.upper()])
            except KeyError:
                atom_types.append(self._atom2int["ANY"])
        return atom_types

    def get_bond_types(self):
        bond_types = dict()
        for bond in self.mol_parser.get_bond_attributes():
            key = bond.origin + "-" + bond.target
            key2 = bond.target + "-" + bond.origin
            try:
                bond_types[key] = bond_types[key2] = \
                    self._bond2int[bond.type.upper()]
            except KeyError:
                bond_types[key] = bond_types[key2] = \
                    self._bond2int["ANY"]
        return bond_types


if __name__ == "__main__":
    from time import sleep
    molGraph = Mol2toGraph(
        r"..\..\data\docking\akt1\active\3cqw-actives34_pose1.mol2")
    print("Bonds:", molGraph.bonds())
    print("Coordinates:", molGraph.atom_coordinates)
    print("Adjacency matrix:", molGraph.get_adjacency_matrix())
    print("Center:", molGraph._find_center())
    print("Surf noms:", molGraph.get_surf_norms())
    print("n_atoms:", molGraph.n_atoms)
    print("atom types:", molGraph.get_atom_types())
    print("bond types:", molGraph.get_bond_types())
    # print(mol_parser.contents)
    # sleep(5)
    # print(mol_parser.content_dict)
    # sleep(5)
    # print(mol_parser.get_atom_attributes())
    # sleep(5)
    # print(mol_parser.get_bond_attributes())
