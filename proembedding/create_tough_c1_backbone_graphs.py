import os
import logging

from chemreader.readers import PDBBB
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


class _Base(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self, verbose=True):
        data_list = []
        raw_path = os.path.join(self.root, "raw")
        files = os.scandir(raw_path)
        pb = tqdm(files) if verbose else files
        for file_ in pb:
            if not file_.name.endswith(".pdb"):
                continue
            try:
                x, edge_idx = self._graph_helper(file_.path)
                data_list.append(Data(x=x, edge_index=edge_idx))
            except AttributeError:
                continue
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _graph_helper(self, fpath):
        graph = PDBBB(fpath).to_graph(sparse=True)
        x = torch.tensor([feat for feat in graph["atom_features"]], dtype=torch.float)
        edge_idx = graph["adjacency"].tocoo()
        edge_idx = torch.tensor([edge_idx.row, edge_idx.col], dtype=torch.long)
        return x, edge_idx


class Control(_Base):
    def __init__(
        self, root="data/tough_c1/control", transform=None, pre_transform=None
    ):
        super().__init__(root, transform, pre_transform)

    def process(self):
        super().process()


class HEME(_Base):
    def __init__(self, root="data/tough_c1/heme", transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    def process(self):
        super().process()


class Steroid(_Base):
    def __init__(
        self, root="data/tough_c1/steroid", transform=None, pre_transform=None
    ):
        super().__init__(root, transform, pre_transform)

    def process(self):
        super().process()


class Nucleotide(_Base):
    def __init__(
        self, root="data/tough_c1/nucleotide", transform=None, pre_transform=None
    ):
        super().__init__(root, transform, pre_transform)

    def process(self):
        super().process()


class PDBBBPocket(PDBBB):

    pockets = {}

    def __init__(self, pdb_fpath, sanitize=True):
        self._pdb_id = os.path.basename(pdb_fpath).split(".")[0][:4].upper()
        super().__init__(fpath=pdb_fpath, sanitize=sanitize)

    def _get_backbone_atoms(self):
        atoms = list()
        for atom in self.rdkit_mol.GetAtoms():
            pdbinfo = atom.GetPDBResidueInfo()
            required_atom = pdbinfo.GetName().strip().upper() in ["N", "CA", "C"]
            try:
                pocket_residue = (
                    pdbinfo.GetResidueNumber() in self.pockets[self._pdb_id]
                )
            except KeyError:
                continue
            if required_atom and pocket_residue:
                atoms.append(atom.GetIdx())
        return atoms

    @classmethod
    def init_pockets(cls, path):
        cls.pockets = dict()
        f = open(path, "r")
        line = f.readline()
        while line:
            tokens = line.strip().split()
            pdb_id = tokens[0][:4].upper()
            residue_indices = [int(idx[1:]) for idx in tokens[4:]]
            cls.pockets[pdb_id] = residue_indices
            line = f.readline()
        f.close()


class PocketDatasets(_Base):
    def __init__(self, root, pocket_fname, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.pocket_fname = pocket_fname
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self, label_value=0, verbose=True):
        data_list = []
        raw_path = os.path.join(self.root, "raw")
        files = os.scandir(raw_path)
        PDBBBPocket.init_pockets(os.path.join(self.root, "raw", self.pocket_fname))
        pb = tqdm(files) if verbose else files
        for file_ in pb:
            if not file_.name.endswith(".pdb"):
                continue
            try:
                x, edge_idx = self._graph_helper(file_.path)
                y = torch.tensor([label_value], dtype=torch.long)
                data_list.append(Data(x=x, edge_index=edge_idx, y=y))
            except AttributeError:
                logging.debug(f"{file_.name} is not valid.")
                continue
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _graph_helper(self, fpath):
        graph = PDBBBPocket(fpath).to_graph(sparse=True)
        x = torch.tensor([feat for feat in graph["atom_features"]], dtype=torch.float)
        edge_idx = graph["adjacency"].tocoo()
        edge_idx = torch.tensor([edge_idx.row, edge_idx.col], dtype=torch.long)
        return x, edge_idx


class ControlPocket(PocketDatasets):
    def __init__(
        self,
        root="data/tough_c1/control_pocket",
        pocket_fname="fpocket-control.residues",
        label_value=0,
        transform=None,
        pre_transform=None,
    ):
        self.pocket_fname = pocket_fname
        super().__init__(root, transform, pre_transform)

    def process(self):
        super().process(label_value=0)


class HEMEPocket(PocketDatasets):
    def __init__(
        self,
        root="data/tough_c1/heme_pocket",
        pocket_fname="fpocket-heme.residues",
        label_value=1,
        transform=None,
        pre_transform=None,
    ):
        self.pocket_fname = pocket_fname
        super().__init__(root, transform, pre_transform)

    def process(self):
        super().process(label_value=1)


class NucleotidePocket(PocketDatasets):
    def __init__(
        self,
        root="data/tough_c1/nucleotide_pocket",
        pocket_fname="fpocket-nucleotide.residues",
        label_value=2,
        transform=None,
        pre_transform=None,
    ):
        self.pocket_fname = pocket_fname
        super().__init__(root, transform, pre_transform)

    def process(self):
        super().process(label_value=2)


class SteroidPocket(PocketDatasets):
    def __init__(
        self,
        root="data/tough_c1/steroid_pocket",
        pocket_fname="fpocket-steroid.residues",
        label_value=3,
        transform=None,
        pre_transform=None,
    ):
        self.pocket_fname = pocket_fname
        super().__init__(root, transform, pre_transform)

    def process(self):
        super().process(label_value=3)
