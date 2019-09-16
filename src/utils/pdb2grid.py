import os
import math
from collections import namedtuple

import numpy as np

from read_pdb import PDB


class PDB2Grid:

    def __init__(self, pdb_f: str):
        self.pdb = PDB(pdb_f)


    @staticmethod
    def density_function(radius):
        if radius <= 2:
            return math.exp(-radius**2/2)
        else:
            return 0.0

    def create_grid(self, dimension, channel=1):
        return np.array(
            (dimension, dimension,dimension, channel), type=np.float16)

    def put_atom_to_grid(self,
                         coordinates,
                         dimension,
                         channel=0,
                         cutoff=2,
                         power=2):
        start = int(dimension//2)
        grid = np.zeros((dimension, dimension, dimension, 3), dtype=np.float16)
        for i in range(dimension):
            for j in range(dimension):
                for k in range(dimension):
                    grid[i][j][k] = np.array([i, j, k])
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

    def get_ca_grid(self, dimension=33):
        grid = np.zeros((dimension, dimension, dimension), dtype=np.float16)
        cas = self.pdb.get_ca()
        normed_coor = self.normalize(cas)
        for idx in range(normed_coor.shape[0]):
            coor = normed_coor[idx]
            atom_in_grid = self.put_atom_to_grid(coor, dimension)
            grid += atom_in_grid
        return grid
    
    def get_ca_res_grid(self, dimension=33):
        pass

    def get_all_atom_grid(self, dimension=33):
        pass

    def get_pocket_ca_grid(self, pocket_residues: list, dimension=33):
        pass

    def get_pocket_ca_res_grid(self, pocket_residues: list, dimension=33):
        pass

    def get_pocket_all_atom_grid(self, pocket_residues: list, dimension=33):
        pass

    def test(self):
        return self.density_function(1.5)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    p = PDB2Grid("../../data/Enzyme/EC_PDB/2rig.pdb")
    ca_grid = p.get_ca_grid()
    z,x,y = ca_grid.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red', marker=".")
    plt.show()