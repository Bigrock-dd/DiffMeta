from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, download_url
import pandas as pd

import utils
from analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges, compute_molecular_metrics
from dataset.abstract_dataset1 import AbstractDatasetInfos, MolecularDataModule


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]
atom_decoder = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']

class MOSESDataset(InMemoryDataset):

    def __init__(self, stage, root, filter_dataset=False, transform=None, pre_transform=None, pre_filter=None):
        self.stage = stage
        self.atom_decoder = atom_decoder
        self.filter_dataset = filter_dataset
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['guacamol_train.csv', 'guacamol_val.csv', 'guacamol_test.csv']

    @property
    def split_file_name(self):
        return ['guacamol_train.csv', 'guacamol_val.csv', 'guacamol_test.csv']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.filter_dataset:
            return ['train_filtered.pt', 'test_filtered.pt', 'test_scaffold_filtered.pt']
        else:
            return ['train.pt', 'test.pt', 'test_scaffold.pt']

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        types = {atom: i for i, atom in enumerate(self.atom_decoder)}

        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        path = self.split_paths[self.file_idx]
        smiles_list = pd.read_csv(path)['smiles'].values

        data_list = []
        smiles_kept = []

        for i, smile in enumerate(tqdm(smiles_list)):
            mol = Chem.MolFromSmiles(smile)
            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            if len(row) == 0:
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            y = torch.zeros(size=(1, 0), dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if self.filter_dataset:
                # Try to build the molecule again from the graph. If it fails, do not add it to the training set
                dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
                dense_data = dense_data.mask(node_mask, collapse=True)
                X, E = dense_data.X, dense_data.E

                assert X.size(0) == 1
                atom_types = X[0]
                edge_types = E[0]
                mol = build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder)
                smiles = mol2smiles(mol)
                if smiles is not None:
                    try:
                        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                        if len(mol_frags) == 1:
                            data_list.append(data)
                            smiles_kept.append(smiles)

                    except Chem.rdchem.AtomValenceException:
                        print("Valence error in GetmolFrags")
                    except Chem.rdchem.KekulizeException:
                        print("Can't kekulize molecule")
            else:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

        if self.filter_dataset:
            smiles_save_path = osp.join(pathlib.Path(self.raw_paths[0]).parent, f'new_{self.stage}.smiles')
            print(smiles_save_path)
            with open(smiles_save_path, 'w') as f:
                f.writelines('%s\n' % s for s in smiles_kept)
            print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")



class MosesDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.remove_h = False
        self.datadir = cfg.dataset.datadir
        self.filter_dataset = cfg.dataset.filter
        self.train_smiles = []
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': MOSESDataset(stage='train', root=root_path, filter_dataset=self.filter_dataset),
                    'val': MOSESDataset(stage='val', root=root_path, filter_dataset=self.filter_dataset),
                    'test': MOSESDataset(stage='test', root=root_path, filter_dataset=self.filter_dataset)}
        super().__init__(cfg, datasets)


class MOSESinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False, meta=None):
        self.name = 'MOSES'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = False
        # 更新后的原子列表，包含了 GuacaMol 数据集中出现的所有元素
        self.atom_decoder = atom_decoder
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        # 对应每个原子的平均原子量（单位：amu），可根据需要进一步调整
        self.num_atom_types = 12
        self.valencies = [4, 3, 2, 1, 3, 1, 1, 1, 3, 2, 2, 4]
        meta_files = dict(n_nodes=f'{self.name}_n_counts.txt',
                          node_types=f'{self.name}_atom_types.txt',
                          edge_types=f'{self.name}_edge_types.txt',
                          valency_distribution=f'{self.name}_valencies.txt')
        
        self.atom_weights = {1: 12, 2: 14, 3: 16, 4: 19, 5: 10.81, 6: 79.9,
                             7: 35.45, 8: 126.9, 9: 30.97, 10: 30.07, 11: 78.97, 12: 28.09}
        self.max_weight = 1000

        self.node_types = torch.tensor([7.4090e-01, 1.0693e-01, 1.1220e-01, 1.4213e-02, 6.0579e-05, 1.7171e-03,
        8.4113e-03, 2.2902e-04, 5.6947e-04, 1.4673e-02, 4.1532e-05, 5.3416e-05])

        self.edge_types = torch.tensor([9.2526e-01, 3.6241e-02, 4.8489e-03, 1.6513e-04, 3.3489e-02])

        self.n_nodes = torch.tensor([0, 0, 3.5760e-06, 2.7893e-05, 6.9374e-05, 1.6020e-04,
                                     2.8036e-04, 4.3484e-04, 7.3022e-04, 1.1722e-03, 1.7830e-03, 2.8129e-03,
                                     4.0981e-03, 5.5421e-03, 7.9645e-03, 1.0824e-02, 1.4459e-02, 1.8818e-02,
                                     2.3961e-02, 2.9558e-02, 3.6324e-02, 4.1931e-02, 4.8105e-02, 5.2316e-02,
                                     5.6601e-02, 5.7483e-02, 5.6685e-02, 5.2317e-02, 5.2107e-02, 4.9651e-02,
                                     4.8100e-02, 4.4363e-02, 4.0704e-02, 3.5719e-02, 3.1685e-02, 2.6821e-02,
                                     2.2542e-02, 1.8591e-02, 1.6114e-02, 1.3399e-02, 1.1543e-02, 9.6116e-03,
                                     8.4744e-03, 6.9532e-03, 6.2001e-03, 4.9921e-03, 4.4378e-03, 3.5803e-03,
                                     3.3078e-03, 2.7085e-03, 2.6784e-03, 2.2050e-03, 2.0533e-03, 1.5598e-03,
                                     1.5177e-03, 9.8626e-04, 8.6396e-04, 5.6429e-04, 5.0422e-04, 2.9323e-04,
                                     2.2243e-04, 9.8697e-05, 9.9413e-05, 6.0077e-05, 6.9374e-05, 3.0754e-05,
                                     3.5045e-05, 1.6450e-05, 2.1456e-05, 1.2874e-05, 1.2158e-05, 5.7216e-06,
                                     7.1520e-06, 2.8608e-06, 2.8608e-06, 7.1520e-07, 2.8608e-06, 1.4304e-06,
                                     7.1520e-07, 0.0000e+00, 0.0000e+00, 0.0000e+00, 7.1520e-07, 0.0000e+00,
                                     1.4304e-06, 7.1520e-07, 7.1520e-07, 0.0000e+00, 1.4304e-06])

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        self.valency_distribution = torch.zeros(self.max_n_nodes * 3 - 2)
        self.valency_distribution[0: 7] = torch.tensor([0.0000, 0.1105, 0.2645, 0.3599, 0.2552, 0.0046, 0.0053])
        if meta is None:
            meta = dict(n_nodes=None, node_types=None, edge_types=None, valency_distribution=None)
        assert set(meta.keys()) == set(meta_files.keys())
        for k, v in meta_files.items():
            if (k not in meta or meta[k] is None) and os.path.exists(v):
                meta[k] = np.loadtxt(v)
                setattr(self, k, meta[k])
        if recompute_statistics or self.n_nodes is None:
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(meta_files["n_nodes"], self.n_nodes.numpy())
            self.max_n_nodes = len(self.n_nodes) - 1
        if recompute_statistics or self.node_types is None:
            self.node_types = datamodule.node_types()  # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt(meta_files["node_types"], self.node_types.numpy())
        if recompute_statistics or self.edge_types is None:
            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(meta_files["edge_types"], self.edge_types.numpy())
        if recompute_statistics or self.valency_distribution is None:
            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt(meta_files["valency_distribution"], valencies.numpy())
            self.valency_distribution = valencies
        # 完成数据信息补全
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        self.train_smiles = []


def get_train_smiles(cfg, datamodule, dataset_infos, evaluate_dataset=False):
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_path = os.path.join(base_path, cfg.dataset.datadir)

    train_smiles = None
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.array(open(smiles_path).readlines())

    if evaluate_dataset:
        train_dataloader = datamodule.dataloaders['train']
        all_molecules = []
        for i, data in enumerate(tqdm(train_dataloader)):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles


if __name__ == "__main__":
    ds = [MOSESDataset(s, "./dataset/guacamol/") for s in ["train", "val", "test"]]