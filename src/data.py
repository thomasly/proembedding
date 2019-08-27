import os
from collections import namedtuple
import tarfile
import random
import abc
import pickle as pk

import numpy as np
import pandas as pd
from tqdm import tqdm

class TOUGH_C1:

    def __init__(self, random_seed=0, train_test_ratio=0.9):
        # root path of the TOUGH-C1 data
        self.root = os.path.join(os.path.pardir, "data", "osfstorage-archive")
        self.bin = os.path.join(os.path.pardir, "bin")
        # dict for label generation
        self.label_dict = {
            "control": 0,
            "heme": 1,
            "nucleotide": 2,
            "steroid": 3
        }
        # load the pdb id lists of the subsets
        self.control_ls = self._load_lst("deepdrug3d_control.lst")
        self.heme_ls = self._load_lst("deepdrug3d_heme.lst")
        self.nucleotide_ls = self._load_lst("deepdrug3d_nucleotide.lst")
        self.steroid_ls = self._load_lst("deepdrug3d_steroid.lst")
        # create training and testing sets
        self.random_seed = random_seed
        self.train_ls, self.test_ls = self._split_dataset(train_test_ratio)
        self.dataset = None
        self.resiName2int = dict()
        self.atomName2int = dict()
        
    def _load_lst(self, path):
        """ Load pdb file list from the lst files
        """
        file_path = os.path.join(self.root, path)
        lb = self.label_dict[path.split(".")[0].split("_")[1]]
        Protein = namedtuple("protein", "label id")
        ret_ls = list()
        with open(file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            ret_ls.append(Protein(label=lb, id=line))
        return ret_ls

    def _load_tar(self, path):
        """ Load pdb files from the .tar.gz files
        """
        tar = tarfile.open(os.path.join(self.root, path))
        pdb_info = dict()
        Atom = namedtuple(
                    "atom",
                    "atom_id atom_name residue_id residue_name x y z"
                )
        resiNameIndex = len(self.resiName2int)
        atomNameIndex = len(self.atomName2int)
        print(f"loading {path} ...")
        for pdb in tqdm(tar.getmembers()):
            if not pdb.isfile():
                continue
            if pdb.name.split("/")[1].startswith("."):
                continue
            name = pdb.name.split("/")[1].split(".")[0]
            try:
                assert(len(name)==5)
            except AssertionError:
                print(f"{pdb.name} is not a valid pdb file")
                continue
            atom_ls = list()
            pdb_f = tar.extractfile(pdb)
            lines = pdb_f.readlines()
            pdb_f.close()
            for line in lines:
                if not line.startswith(b"ATOM"):
                    continue
                line = line.decode("utf-8")
                atom_id = int(line[6:11])
                atom_name = line[12:16].strip().upper()
                try:
                    atom_name = self.atomName2int[atom_name]
                except KeyError:
                    self.atomName2int[atom_name] = atomNameIndex
                    atom_name = atomNameIndex
                    atomNameIndex += 1
                residue_id = int(line[22:26])
                residue_name = line[17:20].strip().upper()
                try:
                    residue_name = self.resiName2int[residue_name]
                except KeyError:
                    self.resiName2int[residue_name] = resiNameIndex
                    residue_name = resiNameIndex
                    resiNameIndex += 1
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom = Atom(
                    atom_id, atom_name, residue_id, residue_name, x, y, z)
                atom_ls.append(atom)
            atom_df = pd.DataFrame(atom_ls, columns=Atom._fields)
            atom_df.x = atom_df.x - atom_df.x.mean()
            atom_df.y = atom_df.y - atom_df.y.mean()
            atom_df.z = atom_df.z - atom_df.z.mean()
            pdb_info[name] = atom_df
        
        return pdb_info

    def _split_dataset(self, train_test_ratio):
        """ Split dataset into training and testing sets
        """
        random.seed(self.random_seed)
        random.shuffle(self.control_ls)
        random.shuffle(self.heme_ls)
        random.shuffle(self.nucleotide_ls)
        random.shuffle(self.steroid_ls)

        train_ls = list()
        test_ls = list()
        self._list_extend(train_ls, test_ls, self.control_ls, train_test_ratio)
        self._list_extend(train_ls, test_ls, self.heme_ls, train_test_ratio)
        self._list_extend(
            train_ls, test_ls, self.nucleotide_ls, train_test_ratio)
        self._list_extend(train_ls, test_ls, self.steroid_ls, train_test_ratio)
        random.shuffle(train_ls)
        random.shuffle(test_ls)
        return train_ls, test_ls

    def _list_extend(self, train_ls, test_ls, target_ls, ratio):
        """ Helper function extending the target_ls with the new train_ls and 
            test_ls
        """
        splitor = int(len(target_ls)*ratio)
        train_ls += target_ls[:splitor]
        test_ls += target_ls[splitor:]

    def _load_dataset(self):
        """ Load the whole dataset from .tar.gz files
        """
        pointnet_data_path = os.path.join(self.bin, "pointnet_data")
        try:
            with open(pointnet_data_path, "rb") as f:
                self.dataset = pk.load(f)
        except FileNotFoundError:
            self.dataset = dict()
            for p in ["protein-control.tar.gz",
                    "protein-heme.tar.gz",
                    "protein-nucleotide.tar.gz",
                    "protein-steroid.tar.gz"]:
                self.dataset.update(self._load_tar(p))
            with open(pointnet_data_path, "wb") as f:
                pk.dump(self.dataset, f)


class BatchGenerator:
    """ Batch generator base class. Implement the _create_batch abstract method
        to generate different data types.
    """

    def __init__(self, data_ls, dataset, batch_size, if_shuffle=True):
        self.data_ls = data_ls
        self.dataset = dataset
        self.if_shuffle = if_shuffle
        self.batch_size = batch_size
        self.n_batch = int(len(data_ls)/batch_size)
        
    def __next__(self):
        try:
            self.counter
        except AttributeError:
            self.counter = 0
        if self.counter < self.n_batch:
            self.counter += 1
            return self._create_batch()
        else:
            self._reset()
            return self._create_batch()

    @abc.abstractmethod
    def _create_batch(self):
        raise NotImplementedError

    def __iter__(self):
        self.counter = 0
        if self.if_shuffle:
            random.shuffle(self.data_ls)
        return self

    def _reset(self):
        self.counter = 0
        if self.if_shuffle:
            random.shuffle(self.data_ls)


class PointnetData(BatchGenerator):
    """ Generate pointcloud data for PointNet architecture.
    """

    def __init__(self,
                 data_ls,
                 dataset,
                 batch_size,
                 pointcloud_len,
                 if_shuffle=True,
                 point_channels=3,
                 label_len=1,
                 resi_name_channel=False):
        super(PointnetData, self).__init__(
            data_ls, dataset, batch_size, if_shuffle)
        self.pointcloud_len = pointcloud_len
        self.point_channels = point_channels
        if resi_name_channel:
            self.point_channels += 1
        self.resi_name_channel = resi_name_channel
        self.label_len = label_len

    def _create_batch(self):
        """ Implementing _create_batch method
        """
        batch_indices = range(
            self.counter*self.batch_size, min(
                (self.counter+1)*self.batch_size, len(self.data_ls)))
        point_sets = np.zeros(
            shape=(
                self.batch_size, self.pointcloud_len, self.point_channels, 1),
            dtype=np.float32
        )
        labels = np.zeros(
            shape=(self.batch_size, self.label_len), dtype=np.int8)
        sample_counter = 0
        for i in batch_indices:
            sample_id = self.data_ls[i].id
            label = self.data_ls[i].label
            data = self.dataset[sample_id]
            pointcloud = np.zeros(
                (self.pointcloud_len, self.point_channels, 1),
                dtype=np.float32)
            if self.resi_name_channel:
                pointcloud[:data.loc[data.atom_name==1].shape[0]] = \
                    np.expand_dims(
                        data.loc[data.atom_name==1][[
                            "x", "y", "z", "residue_name"]].values, -1)
            else:
                pointcloud[:data.loc[data.atom_name==1].shape[0]] = \
                    np.expand_dims(
                        data.loc[data.atom_name==1][[
                            "x", "y", "z"]].values, -1)
            point_sets[sample_counter] = pointcloud
            labels[sample_counter] = label
            sample_counter += 1
        return point_sets, labels


class TOUGH_Point(TOUGH_C1):

    def __init__(self,
                 batch_size=32,
                 pointcloud_len=1024,
                 random_seed=0,
                 train_test_ratio=0.9,
                 resi_name_channel=False):
        super(TOUGH_Point, self).__init__(random_seed, train_test_ratio)
        self.batch_size = batch_size
        self.pointcloud_len = pointcloud_len
        self.resi_name_channel = resi_name_channel

    def train(self):
        if self.dataset is None:
            self._load_dataset()
        train_set = PointnetData(
            self.train_ls, self.dataset, self.batch_size,
            self.pointcloud_len, resi_name_channel=self.resi_name_channel)
        self._train_steps = train_set.n_batch
        for point_sets, labels in train_set:
            yield (point_sets, labels)

    def test(self):
        if self.dataset is None:
            self._load_dataset()
        test_set = PointnetData(
            self.test_ls, self.dataset, self.batch_size,
            self.pointcloud_len, if_shuffle=False,
            resi_name_channel=self.resi_name_channel
        )
        self._test_steps = test_set.n_batch
        for point_sets, labels in test_set:
            yield (point_sets, labels)

    @property
    def train_steps(self):
        try:
            return self._train_steps
        except AttributeError:
            next(self.train())
            return self._train_steps

    @property
    def test_steps(self):
        try:
            return self._test_steps
        except AttributeError:
            next(self.test())
            return self._test_steps


class TOUGH_POCKET()

if __name__ == "__main__":
    # def print_first(d):
    #     key = list(d.keys())[0]
    #     print(key)
    #     print(d[key][0:10])

    # t = TOUGH_C1()
    # print(f"control: {t.control_ls[0:2]}")
    # print(f"heme: {t.heme_ls[0:2]}")
    # print(f"nucleotide: {t.nucleotide_ls[0:2]}")
    # print(f"steroid: {t.steroid_ls[0:2]}")
    # print_first(t._load_tar("protein-control.tar.gz"))
    # print(len(t.train_ls))
    # print(len(t.test_ls))
    # print(t.train_ls[:10])
    # print(t.test_ls[:10])
    # print(len(t.control_ls))
    # print(len(t.heme_ls))
    # print(len(t.nucleotide_ls))
    # print(len(t.steroid_ls))

    tp = TOUGH_Point(resi_name_channel=True)
    # train, train_lb = next(tp.train())
    # test, test_lb = next(tp.test())
    # print("train shape:", train.shape)
    # print("train label shape:", train_lb.shape)
    # print("train avg:", np.mean(train, axis=1))

    # print("test shape:", test.shape)
    # print("test label shape:", test_lb.shape)
    # print("test avg:", np.mean(test, axis=1))

    # print("train set sample:", train[0][:10])
    # print("test set sample:", test[0][:10])
    for i in tp.train():
        print(i[0].shape)

