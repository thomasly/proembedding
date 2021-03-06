from collections import namedtuple
import copy

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA


class Mol2Parser:

    def __init__(self, mol2_path):
        self._path = mol2_path

    @property
    def molecule_name(self):
        try:
            return self._molecule_name
        except AttributeError:
            self._molecule_name = self.content_dict["MOLECULE"][0]
            return self._molecule_name

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
                  "O.3,O.2,O.co2,O.spc,O.t3p,S.3,S.2,S.O,S.O2,P.3,F,Cl,Br,I,"\
                  "H,H.spc,H.t3p,LP,Du,Du.C,Hal,Het,Hev,Li,Na,Mg,Al,Si,K,Ca,"\
                  "Cr.thm,Cr.oh,Mn,Fe,Co.oh,Cu,Any".split(",")
    _atom2int = {atom.upper(): idx for idx, atom in enumerate(_atom_types)}

    _bond_types = "1,2,3,am,ar,du,un,cn,any".split(",")
    _bond2int = {bond.upper(): idx for idx, bond in enumerate(_bond_types)}

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
    def atom_charges(self):
        try:
            return self._atom_charges
        except AttributeError:
            self._atom_charges = list()
            for atom in self.mol_parser.get_atom_attributes():
                self._atom_charges.append(float(atom.charge))
        return self._atom_charges

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

    @property
    def mol_name(self):
        try:
            return self._mol_name
        except AttributeError:
            self._mol_name = self.mol_parser.molecule_name
            return self._mol_name


class Mol2toGrid:

    _atom_types = "C,N,O,S,P,B,F,Cl,Br,I,H,Any".split(",")
    _atom2int = {atom.upper(): idx for idx, atom in enumerate(_atom_types)}

    def __init__(self, mol2_path):
        self._path = mol2_path
        self.mol_parser = Mol2Parser(mol2_path)

    def _initialize_channel_grid(self, dimension, channels=20):
        grid = np.zeros(
            (channels, dimension, dimension, dimension),
            dtype=np.float16)
        return grid

    def _get_coodinate_grid(self, dimension, zoom, n_coors=3):
        coo_grid = np.zeros(
            (dimension, dimension, dimension, n_coors), dtype=np.float16)
        for i in range(dimension):
            for j in range(dimension):
                for k in range(dimension):
                    coo_grid[i][j][k] = np.array([i/zoom, j/zoom, k/zoom])
        return coo_grid

    def coo_grid(self, dimension, zoom):
        try:
            return self._coo_grid
        except AttributeError:
            self._coo_grid = self._get_coodinate_grid(dimension, zoom=zoom)
            return self._coo_grid

    def _parse_atoms(self, atoms):
        atom_coos, atom_types = list(), list()
        for atom in atoms:
            atom_coos.append(list(map(float, [atom.x, atom.y, atom.z])))
            atom_name = atom.type.split(".")[0].upper()
            if atom_name == "DU":
                atom_name = "C"
            try:
                atom_type = self._atom2int[atom_name]
            except KeyError:
                atom_type = self._atom2int["ANY"]
            atom_types.append(atom_type)
        return atom_coos, atom_types

    def normalize(self, atoms):
        normed_coor = np.array(atoms)
        avgs = np.mean(normed_coor, axis=0)
        return normed_coor - avgs

    def rotate_norm(self, atoms):
        rotate_norm_coor = np.array(atoms)
        pca = PCA(n_components=3)
        pca.fit(atoms)
        rotate_norm_coor = pca.transform(rotate_norm_coor)
        avgs = np.mean(rotate_norm_coor, axis=0)
        return rotate_norm_coor - avgs

    def put_atom_to_grid(self,
                         coordinates,
                         dimension,
                         zoom,
                         cutoff,
                         power):
        start = int(dimension/zoom//2)
        grid = copy.deepcopy(self.coo_grid(dimension, zoom=zoom))
        grid = grid - start
        distances = np.linalg.norm((grid - coordinates), axis=3)
        atom_rep = np.where(distances < cutoff, distances, float("inf"))
        return np.exp(-np.power(atom_rep, power)/2)

    def get_grid(self, dimension=33, zoom=2, rotate=False, cutoff=2, power=2):
        channels = len(self._atom_types)
        grid = self._initialize_channel_grid(dimension, channels=channels)
        atoms = self.mol_parser.get_atom_attributes()
        atom_coos, atom_types = self._parse_atoms(atoms)
        if rotate:
            atom_coos = self.rotate_norm(atom_coos)
        else:
            atom_coos = self.normalize(atom_coos)
        for coo, typ in zip(atom_coos, atom_types):
            atom_in_grid = self.put_atom_to_grid(
                coo, dimension, zoom=zoom, cutoff=cutoff, power=power)
            grid[typ] += atom_in_grid
        return grid


class Mol2toPointCloud:

    _atom_types = "C.3,C.2,C.1,C.ar,C.cat,N.3,N.2,N.1,N.ar,N.am,N.pl3,N.4,"\
                  "O.3,O.2,O.co2,O.spc,O.t3p,S.3,S.2,S.O,S.O2,P.3,F,Cl,Br,I,"\
                  "H,H.spc,H.t3p,LP,Du,Du.C,Hal,Het,Hev,Li,Na,Mg,Al,Si,K,Ca,"\
                  "Cr.thm,Cr.oh,Mn,Fe,Co.oh,Cu,Any".split(",")
    _atom2int = {atom.upper(): idx for idx, atom in enumerate(_atom_types)}

    def __init__(self, mol2_path):
        self._path = mol2_path
        self.mol_parser = Mol2Parser(mol2_path)

    def normalize(self, point_cloud):
        avgs = np.mean(point_cloud, axis=0)
        return point_cloud - avgs

    def get_point_cloud(self, include_type=True):
        # get the attributes of atoms from the .mol2 file
        atoms = self.mol_parser.get_atom_attributes()

        # initialize the point cloud
        point_cloud = np.zeros((len(atoms), 3))
        type_channel = np.zeros((len(atoms), 1))
        # save coordinates and atom type into the point cloud
        for i, atom in enumerate(atoms):
            point_cloud[i] = np.array([atom.x, atom.y, atom.z])
            try:
                type_channel[i] = self._atom2int[atom.type.upper()]
            except KeyError:
                type_channel[i] = self._atom2int["ANY"]
        point_cloud = self.normalize(point_cloud)
        if include_type:
            point_cloud = np.concatenate([point_cloud, type_channel], axis=1)
        return point_cloud


if __name__ == "__main__":
    # from time import sleep
    # molGraph = Mol2toGraph(
    #     r"..\..\data\docking\akt1\active\3cqw-actives34_pose1.mol2")
    # print("Bonds:", molGraph.bonds())
    # print("Coordinates:", molGraph.atom_coordinates)
    # print("Adjacency matrix:", molGraph.get_adjacency_matrix())
    # print("Center:", molGraph._find_center())
    # print("Surf noms:", molGraph.get_surf_norms())
    # print("n_atoms:", molGraph.n_atoms)
    # print("atom types:", molGraph.get_atom_types())
    # print("bond types:", molGraph.get_bond_types())
    # print(mol_parser.contents)
    # sleep(5)
    # print(mol_parser.content_dict)
    # sleep(5)
    # print(mol_parser.get_atom_attributes())
    # sleep(5)
    # print(mol_parser.get_bond_attributes())
    m2pc = Mol2toPointCloud(
        r"..\..\data\docking\akt1\actives\3cqw-actives100_pose1.mol2")
    cloud = m2pc.get_point_cloud(include_type=True)
    print(cloud)
    print("shape:", cloud.shape)
