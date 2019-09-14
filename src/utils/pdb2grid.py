import os
import math

import numpy as np


class PDB2Grid:

    def __init__(self, pdb_f: str):
        self.pdb_f = pdb_f

    @staticmethod
    def density_function(radius):
        if radius <= 2:
            return math.exp(-radius**2/2)
        else:
            return 0.0

    def create_grid(self, dimension, channel=1):
        return np.array(
            (dimension, dimension,dimension, channel), type=np.float16)

    def put_atom_to_grid(self, coordinates, dimension, channel=0, cutoff=2):
        start = int(dimension//2)
        grid = np.zeros((dimension, dimension, dimension, 3), dtype=np.float16)
        for i in range(dimension):
            for j in range(dimension):
                for k in range(dimension):
                    grid[i][j][k] = np.array([i, j, k])
        grid = grid - start
        distances = np.linalg.norm((grid - coordinates), axis=3)
        atom_rep = np.where(distances<cutoff, distances, float("inf"))
        return np.exp(-np.power(atom_rep, 2)/2)

a = put_atom_to_grid("a", [1, 1, 1], 5)

    def get_ca_grid(self, dimension=32):
        pass
    
    def get_ca_res_grid(self, dimension=32):
        pass

    def get_all_atom_grid(self, dimension=32):
        pass

    def get_pocket_ca_grid(self, pocket_residues: list, dimension=32):
        pass

    def get_pocket_ca_res_grid(self, pocket_residues: list, dimension=32):
        pass

    def get_pocket_all_atom_grid(self, pocket_residues: list, dimension=32):
        pass

    def test(self):
        return self.density_function(1.5)


if __name__ == "__main__":
    p = PDB2Grid("a")
    print(p.test())