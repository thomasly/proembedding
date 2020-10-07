from chemreader.readers.readmol import MolReader, MolBlock
from torch_geometric.data import Data, InMemoryDataset
import torch


class MxmNetDrugs(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["drugs_MXMNET.mol"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        reader = MolReader(self.raw_paths[0])
        data_list = list()
        for block_txt in reader.blocks:
            block = MolBlock(block_txt.strip())
            try:
                print("start")
                graph = block.to_graph(sparse=True)
                print("end")
            except AttributeError:  # block cannot be converted to rdkit Mol
                continue
            x = torch.tensor(graph["atom_features"], dtype=torch.float)
            edge_idx = graph["adjacency"].tocoo()
            edge_idx = torch.tensor([edge_idx.row, edge_idx.col], dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_idx))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
