import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix


class XYZParser:

    def __init__(self, path):
        self.path = path

    @property
    def lines(self):
        try:
            return self._lines
        except AttributeError:
            with open(self.path, "r") as f:
                self._lines = f.readlines()
            return self._lines

    @property
    def n_atoms(self):
        try:
            return self._n_atoms
        except AttributeError:
            self._n_atoms = self._get_atom_number()
            return self._n_atoms

    def _get_atom_number(self):
        return int(self.lines[0])

    @property
    def atoms(self):
        try:
            return self._atoms
        except AttributeError:
            self._atoms = self._get_atoms()
            return self._atoms

    def _get_atoms(self):
        atoms = list()
        for line in self.lines[2:2+self.n_atoms]:
            atom = line.split()[0]
            atoms.append(atom)
        return atoms

    def _asfloat(self, x):
        try:
            return float(x)
        except ValueError:
            return float("e".join(x.split("*^")))

    def get_atom_coordinates(self):
        coordinates = list()
        for line in self.lines[2:2+self.n_atoms]:
            try:
                coor = list(map(self._asfloat, line.split()[1:4]))
                coordinates.append(coor)
            except ValueError:
                print("{} lacks coordinates information.".format(self.path))
                raise
        return coordinates

    def get_atom_charges(self):
        charges = list()
        for line in self.lines[2:2+self.n_atoms]:
            chrg = self._asfloat(line.split()[-1])
            charges.append(chrg)
        return charges

    @property
    def smiles(self):
        try:
            return self._smiles
        except AttributeError:
            self._smiles = self._get_canonical_smiles()
            return self._smiles

    @property
    def b3lyp_smiles(self):
        try:
            return self._b3lyp_smiles
        except AttributeError:
            self._b3lyp_smiles = self._get_b3lyp_smiles()
            return self._b3lyp_smiles

    def _get_smiles(self):
        return self.lines[self.n_atoms+3].stripe().split()

    def _get_canonical_smiles(self):
        return self._get_smiles()[0]

    def _get_b3lyp_smiles(self):
        return self._get_smiles()[1]

    @property
    def comments(self):
        try:
            return self._comments
        except AttributeError:
            self._comments = self._get_comments()
            return self._comments

    def _get_comments(self):
        return self.lines[1].strip()


class XYZ2Graph:

    atom_types = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'F': 4}

    def __init__(self, path):
        self.xyz = XYZParser(path)

    @property
    def n_nodes(self):
        return self.xyz.n_atoms

    def get_adjacency_matrix(self, cutoff=5):
        atom_coor = np.array(self.xyz.get_atom_coordinates())
        distances = cdist(atom_coor, atom_coor)
        mask = np.logical_and(distances < cutoff, distances != 0)
        adjacency_matrix = np.where(mask, 1, 0)
        return coo_matrix(adjacency_matrix)

    def get_surf_norms(self, coordinates):
        return coordinates - np.mean(coordinates, axis=0)

    @property
    def node_attributes(self):
        coordinates = np.array(self.xyz.get_atom_coordinates())
        norms = self.get_surf_norms(coordinates)
        atom_types = np.expand_dims(
            np.array(list(map(self.atom_types.get, self.xyz.atoms))), 1)
        charges = np.expand_dims(np.array(self.xyz.get_atom_charges()), 1)
        attributes = np.concatenate(
            [coordinates, norms, atom_types, charges], axis=1)
        return attributes

    @property
    def graph_label(self):
        return self.xyz.comments

    def get_atom_types(self):
        atom_types = list()
        for atom in self.mol_parser.get_atom_attributes():
            try:
                atom_types.append(self._atom2int[atom.type.upper()])
            except KeyError:
                atom_types.append(self._atom2int["ANY"])
        return atom_types
