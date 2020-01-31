# import os
# from utils.mol2parser import Mol2Parser

# files = os.scandir("../data/docking/akt1/active")
# atoms_types = set()
# for fi in files:
#     if not fi.name.endswith(".mol2"):
#         continue
#     m2p = Mol2Parser(fi.path)
#     atoms = m2p.get_atom_attributes()
#     for a in atoms:
#         atoms_types.add(a.type.split(".")[0])

# print(atoms_types)

#################################################################

import pickle as pk
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
with open("../test/akt1.grids", "rb") as f:
    grid = pk.load(f)
# print("grid:", next(iter(grid.values()))[0, 10:20])
print("Mol:", next(iter(grid.keys())))
dim = 33

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
c_grid = next(iter(grid.values()))[0]
z,x,y = c_grid.nonzero()
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_xlim(0, dim)
ax1.set_ylim(0, dim)
ax1.set_zlim(0, dim)
ax1.scatter(x, y, z, zdir='z', c='red', marker=".")
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.set_xlim(0, 33)
# ax2.set_ylim(0, 33)
# ax2.set_zlim(0, 33)
# ax2.scatter(x2, y2, z2, zdir='z', c='red', marker=".")

plt.show()
