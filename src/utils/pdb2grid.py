import os
import math
import copy
from collections import namedtuple

import numpy as np
from sklearn.decomposition import PCA

from .read_pdb import PDB


class PDB2Grid:

    def __init__(self, pdb_f: str):
        self.pdb = PDB(pdb_f)


    # @staticmethod
    # def density_function(radius):
    #     if radius <= 2:
    #         return math.exp(-radius**2/2)
    #     else:
    #         return 0.0

    def create_grid(self, dimension, channel=1):
        return np.array(
            (dimension, dimension,dimension, channel), type=np.float16)

    def _initialize_grid(self, dimension):
        self.grid = np.zeros(
            (dimension, dimension, dimension, 3), dtype=np.float16)
        for i in range(dimension):
            for j in range(dimension):
                for k in range(dimension):
                    self.grid[i][j][k] = np.array([i, j, k])

    def put_atom_to_grid(self,
                         coordinates,
                         dimension,
                         channel=0,
                         cutoff=2,
                         power=2):
        start = int(dimension//2)
        try:
            grid = copy.deepcopy(self.grid) 
        except AttributeError:
            self._initialize_grid(dimension)
            grid = copy.deepcopy(self.grid)
        grid = grid - start
        distances = np.linalg.norm((grid - coordinates), axis=3)
        atom_rep = np.where(distances < cutoff, distances, float("inf"))
        return np.exp(-np.power(atom_rep, power)/2)

    def normalize(self, atoms):
        normed_coor = np.zeros((len(atoms), 3))
        for idx, atom in enumerate(atoms):
            normed_coor[idx] = [atom.x, atom.y, atom.z]
        avgs = np.mean(normed_coor, axis=0)
        return normed_coor - avgs

    def rotate_norm(self, atoms):
        rotate_norm_coor = np.zeros((len(atoms), 3))
        for idx, atom in enumerate(atoms):
            rotate_norm_coor[idx] = [atom.x, atom.y, atom.z]
        pca = PCA(n_components=3)
        pca.fit(rotate_norm_coor)
        rotate_norm_coor = pca.transform(rotate_norm_coor)
        avgs = np.mean(rotate_norm_coor, axis=0)
        return rotate_norm_coor - avgs

    def get_ca_grid(self, dimension=33):
        grid = np.zeros((dimension, dimension, dimension), dtype=np.float16)
        cas = self.pdb.get_ca()
        normed_coor = self.normalize(cas)
        for idx in range(normed_coor.shape[0]):
            coor = normed_coor[idx]
            atom_in_grid = self.put_atom_to_grid(coor, dimension)
            grid += atom_in_grid
        grid = np.expand_dims(grid, 0)
        return grid
    
    def get_ca_res_grid(self, dimension=33):
        pass

    def get_all_atom_grid(self, dimension=33):
        pass

    def get_pocket_ca_grid(self, pocket_residues: list, dimension=33):
        grid = np.zeros((dimension, dimension, dimension), dtype=np.float16)
        cas = self.pdb.get_ca()
        pocket_ca = list()
        for ca in cas:
            if ca.resi_id in pocket_residues:
                pocket_ca.append(ca)
        normed_coor = self.normalize(pocket_ca)
        for idx in range(normed_coor.shape[0]):
            coor = normed_coor[idx]
            atom_in_grid = self.put_atom_to_grid(coor, dimension)
            grid += atom_in_grid
        grid = np.expand_dims(grid, 0)
        return grid

    def _initialize_res_grid(self, dimension, channels=20):
        grid = np.zeros(
            (channels, dimension, dimension, dimension),
            dtype=np.float16)
        return grid

    def _get_pocket_ca(self, all_cas, pocket_residues):
        pocket_ca = list()
        pocket_resi = list()
        for ca in all_cas:
            if ca.resi_id in pocket_residues:
                pocket_ca.append(ca)
                pocket_resi.append(PDB.resi2int(ca.resi_name))
        return pocket_ca, pocket_resi

    def get_pocket_ca_res_grid(self, pocket_residues: list, dimension=33):
        grid = self._initialize_res_grid(dimension, channels=20)
        cas = self.pdb.get_ca()
        pocket_ca, pocket_resi = self._get_pocket_ca(cas, pocket_residues)
        normed_coor = self.normalize(pocket_ca)
        for idx in range(normed_coor.shape[0]):
            coor = normed_coor[idx]
            atom_in_grid = self.put_atom_to_grid(coor, dimension)
            grid[pocket_resi[idx]] += atom_in_grid
        return grid

    def get_pocket_ca_res_pca_grid(self, pocket_residues: list, dimension=33):
        grid = self._initialize_res_grid(dimension, channels=20)
        cas = self.pdb.get_ca()
        pocket_ca, pocket_resi = self._get_pocket_ca(cas, pocket_residues)
        rotate_norm_grid = self.rotate_norm(pocket_ca)
        for idx in range(rotate_norm_grid.shape[0]):
            coor = rotate_norm_grid[idx]
            atom_in_grid = self.put_atom_to_grid(coor, dimension)
            grid[pocket_resi[idx]] += atom_in_grid
        return grid

    def get_pocket_all_atom_grid(self, pocket_residues: list, dimension=33):
        pass

    def test(self):
        return self.density_function(1.5)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    p = PDB2Grid("../data/tough_c1/protein-control/5c11A.pdb")
    ca_grid = p.get_pocket_ca_res_grid([18, 19, 20, 22, 28, 41, 42, 43, 44, 45, 46])
    rotated_ca_grid = p.get_pocket_ca_res_pca_grid([18, 19, 20, 22, 28, 41, 42, 43, 44, 45, 46])
    z,x,y = ca_grid[2].nonzero()
    z2, x2, y2 = rotated_ca_grid[2].nonzero()
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlim(0, 33)
    ax1.set_ylim(0, 33)
    ax1.set_zlim(0, 33)
    ax1.scatter(x, y, z, zdir='z', c='red', marker=".")
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlim(0, 33)
    ax2.set_ylim(0, 33)
    ax2.set_zlim(0, 33)
    ax2.scatter(x2, y2, z2, zdir='z', c='red', marker=".")

    plt.show()