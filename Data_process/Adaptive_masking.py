import math
import torch
import random
import numpy as np
import pickle as pkl
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import RobertaTokenizerFast
from torch_geometric.data import Data, Batch


def smiles_to_graph(mol, node_features, bond_hidden):
    mol = mol[0]
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    node_features_new = node_features[:mol.GetNumAtoms(), :]

    bond_features_new = np.zeros((mol.GetNumBonds() * 2, bond_hidden))

    for bond in mol.GetBonds():
        bond_features_new[bond.GetIdx() * 2, :] = get_bond_features(bond, bond_hidden)
        bond_features_new[bond.GetIdx() * 2 + 1, :] = get_bond_features(bond, bond_hidden)

    bond_features_new = torch.from_numpy(bond_features_new)

    return Data(x=node_features_new, edge_index=edge_index, edge_attr=bond_features_new)


def one_hot_vector(val, lst, add_unknown=True):
    if add_unknown:
        vec = np.zeros(len(lst) + 1)
    else:
        vec = np.zeros(len(lst))

    vec[lst.index(val) if val in lst else -1] = 1
    return vec


def get_atom_features(atom, atom_hidden, atom_rings=None):
    v1 = one_hot_vector(atom.GetAtomicNum(), [i for i in range(1, 101)])
    v2 = one_hot_vector(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                  Chem.rdchem.HybridizationType.SP2,
                                                  Chem.rdchem.HybridizationType.SP3,
                                                  Chem.rdchem.HybridizationType.SP3D,
                                                  Chem.rdchem.HybridizationType.SP3D2])
    v4 = [
        atom.GetTotalNumHs(includeNeighbors=True) / 8,
        atom.GetDegree() / 4,
        atom.GetFormalCharge() / 8,
        atom.GetTotalValence() / 8,
        0 if math.isnan(atom.GetDoubleProp('_GasteigerCharge')) or math.isinf(
            atom.GetDoubleProp('_GasteigerCharge')) else atom.GetDoubleProp('_GasteigerCharge'),
        0 if math.isnan(atom.GetDoubleProp('_GasteigerHCharge')) or math.isinf(
            atom.GetDoubleProp('_GasteigerHCharge')) else atom.GetDoubleProp('_GasteigerHCharge'),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing())
    ]
    v5 = [atom.GetIdx() + 1]
    attributes = np.concatenate([v1, v2, v4, v5], axis=0)
    assert len(attributes) == atom_hidden + 1
    return attributes


def get_bond_features(bond, bond_hidden, bond_rings=None):
    v1 = one_hot_vector(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE,
                                             Chem.rdchem.BondType.DOUBLE,
                                             Chem.rdchem.BondType.TRIPLE,
                                             Chem.rdchem.BondType.AROMATIC], add_unknown=False)
    v2 = one_hot_vector(bond.GetStereo(), [Chem.rdchem.BondStereo.STEREOANY,
                                           Chem.rdchem.BondStereo.STEREOCIS,
                                           Chem.rdchem.BondStereo.STEREOE,
                                           Chem.rdchem.BondStereo.STEREONONE,
                                           Chem.rdchem.BondStereo.STEREOTRANS,
                                           Chem.rdchem.BondStereo.STEREOZ], add_unknown=False)
    v4 = [
        int(bond.GetIsConjugated()),
        int(bond.GetIsAromatic()),
        int(bond.IsInRing())
    ]
    attributes = np.concatenate([v1, v2, v4])
    assert len(attributes) == bond_hidden
    return attributes


def load_data_from_mol(mol, atom_hidden, bond_hidden, max_length, max_ids, tokenizer):
    mol = mol[0]
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
    Chem.rdmolops.AssignStereochemistryFrom3D(mol)
    AllChem.ComputeGasteigerCharges(mol)
    atom_rings = mol.GetRingInfo().AtomRings()
    node_features = np.array([get_atom_features(atom, atom_hidden, atom_rings) for atom in mol.GetAtoms()])
    bond_rings = mol.GetRingInfo().BondRings()
    bond_features = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms(), bond_hidden))

    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtom().GetIdx()
        end_atom_idx = bond.GetEndAtom().GetIdx()
        bond_features[begin_atom_idx, end_atom_idx, :] = bond_features[end_atom_idx, begin_atom_idx, :] = \
            get_bond_features(bond, bond_hidden)

    adjacency_matrix = Chem.rdmolops.GetDistanceMatrix(mol).astype(np.float64)
    smiles = Chem.MolToSmiles(mol)
    token_ids = tokenizer.encode(smiles)
    if token_ids.__len__() > 60:
        pass
    token_ids = np.array(token_ids)
    token_ids = token_ids.reshape((1, token_ids.shape[0]))

    return pad_array(node_features, (max_length, node_features.shape[-1])), \
        pad_array(bond_features, (max_length, max_length, bond_features.shape[-1])), \
        pad_array(adjacency_matrix, (max_length, max_length)), \
        pad_array(token_ids, (1, 60))


def load_data_from_mol_mask(mol, atom_hidden, bond_hidden, max_length, max_ids, tokenizer):
    node_features, bond_features, adjacency_matrix, token_ids = load_data_from_mol(mol, atom_hidden, bond_hidden,
                                                                                   max_length, max_ids, tokenizer)
    node_features = torch.from_numpy(node_features)
    bond_features = torch.from_numpy(bond_features)
    adjacency_matrix = torch.from_numpy(adjacency_matrix)
    token_ids = torch.from_numpy(token_ids)
    num_atoms = (torch.sum(torch.abs(node_features), dim=-1) != 0).sum().item()

    # Adaptive Priority Masks
    # Base Masking Ratio Based on Molecular Size
    base_mask_ratio = 0.15 if num_atoms < 20 else 0.30

    # Retrieve Ring Information
    mol_obj = mol[0]
    ring_info = mol_obj.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    ring_atoms = set()
    for ring in atom_rings:
        ring_atoms.update(ring)

    # Build Priority Candidate List
    candidate_indices = []
    for atom_idx in range(num_atoms):
        atom = mol_obj.GetAtomWithIdx(atom_idx)
        if atom_idx not in ring_atoms and atom.GetAtomicNum() == 6:
            candidate_indices.extend([atom_idx] * 3)
        elif atom_idx not in ring_atoms:
            candidate_indices.extend([atom_idx] * 2)
        else:
            candidate_indices.append(atom_idx)

    sample_size = int(num_atoms * base_mask_ratio + 1)
    try:
        masked_atom_indices = random.sample(candidate_indices, sample_size)
    except ValueError:
        masked_atom_indices = random.sample(range(num_atoms), sample_size)

    if len(atom_rings) > 0:
        largest_ring = max(atom_rings, key=len)
        if len(largest_ring) >= 6:
            protected_atoms = set(largest_ring)
            masked_atom_indices = [idx for idx in masked_atom_indices if idx not in protected_atoms]


    masked_atom_indices = list(set(masked_atom_indices))[:sample_size]

    if not masked_atom_indices:
        if num_atoms > 0:
            non_ring = [idx for idx in range(num_atoms) if idx not in ring_atoms]
            if non_ring:
                masked_atom_indices = [random.choice(non_ring)]
            else:
                masked_atom_indices = [random.randint(0, num_atoms - 1)]
        else:
            raise ValueError("Invalid molecule with 0 atoms")

    mask_node_labels_list = []
    for atom_idx in masked_atom_indices:
        mask_node_labels_list.append(node_features[atom_idx].view(1, -1))

    if not mask_node_labels_list:
        dummy = torch.zeros(1, node_features.size(-1))
        mask_node_labels_list.append(dummy)
    mask_node_labels = torch.cat(mask_node_labels_list, dim=0)
    masked_atom_indices = torch.tensor(masked_atom_indices)

    for atom_idx in masked_atom_indices:
        node_features[atom_idx] = torch.zeros(1, node_features.size()[-1])

    # SMILES masking
    labels = token_ids.clone()
    token_ids_new = token_ids.clone()
    remove = ~torch.isin(token_ids_new, torch.tensor([12, 13, 17, 18, 20, 21, 22, 26, 31, 32, 38, 43, 58, 76, 98, 124]))
    a = token_ids_new[remove]
    a[masked_atom_indices] = 0
    token_ids_new[remove] = a
    special = torch.isin(token_ids_new, torch.tensor([11, 12, 13, 14, 591, 592]))
    token_ids_new[special] = 0
    smiles_masked_indices = token_ids_new.bool()
    masked_indices = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & smiles_masked_indices
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    token_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    token_ids = token_ids.long()
    token_ids[indices_random] = random_words[indices_random]

    return node_features, bond_features, adjacency_matrix, mask_node_labels, masked_atom_indices, token_ids, labels


def pad_array(array, shape, dtype=np.float32):
    padded_array = np.zeros(shape, dtype=dtype)
    if len(shape) == 2:
        padded_array[:array.shape[0], :array.shape[1]] = array
    elif len(shape) == 3:
        padded_array[:array.shape[0], :array.shape[1], :] = array
    return padded_array


def construct_dataset_mask(mol_list, atom_hidden, bond_hidden, max_length, max_ids, tokenizer):
    for idx, mol in enumerate(tqdm(zip(mol_list), total=len(mol_list))):
        node_features, bond_features, adjacency_matrix, mask_node_labels, masked_atom_indices, token_ids, labels = load_data_from_mol_mask(
            mol, atom_hidden, bond_hidden, max_length, max_ids, tokenizer)
        data = smiles_to_graph(mol, node_features, bond_hidden)
        data_dict = {'node_features': node_features.numpy(),
                     'bond_features': bond_features.numpy(),
                     'adjacency_matrix': adjacency_matrix.numpy(),
                     'mask_node_labels': mask_node_labels.numpy(),
                     'masked_atom_indices': masked_atom_indices.numpy(),
                     'token_ids': token_ids.numpy(),
                     'labels': labels.numpy(),
                     'edge_attr': data.edge_attr.numpy(),
                     'edge_index': data.edge_index.numpy(),
                     'num_atoms': np.array(data.num_nodes),
                     'x': data.x.numpy()
                     }
        np.savez(f'/home/jjiang/xxli/SMFP/data/zinc15/processed_geometric_mol{idx}.npz', **data_dict)
    return 1


if __name__ == '__main__':
    with open('/home/jjiang/xxli/SMFP/data/zinc15_250K.pickle', 'rb') as f:
        mol = pkl.load(f)
    max_length = max([data.GetNumAtoms() for data in mol])
    tokenizer = RobertaTokenizerFast.from_pretrained("/home/jjiang/xxli/SMFP/Data_process/smiles_tokenized_PubChem_shard00_160k/", max_len=512,
                                                     additional_special_tokens = ['[Randic connectivity]','[NH4]','[201Tl]','[ghi]','[(1R)-1-methylpropyl]','[(2S)-butan-2-yl]','[2-benzhydryloxyethyl]','[Pd-3]','[W+2]','[Cu-4]','[Pt-3]','[Mo-3]','[GaH3]','[U-5]','[Tl-]','[V+]','[Zn-2]','[Ca-4]','[Co-4]','[Re+]','[Tl-3]','[GeH2+]','[V-]','[Co-2]','[Pt-4]','[FH+]','[Cu-5]','[Ru-4]','[Zn-4]','[Zn-3]','[Cu-3]','[Ca-2]','[Ga-3]','[Rh-4]','[Hg-2]','[BrH2+]','[MgH2]','[AlH3-]','[Ni-2]','[Ti-2]','[B-2]','[Ni-3]','[Au-3]','[b-]','[Ir-4]','[Pd-4]','[Ni-4]','[AlH3-3]','[Mo-]','[249Cf]','[TlH2+]','[SbH6+3]','[Si-2]','[SnH2+2]','[PbH2+2]','[S@+]','[Po]','[HgH]','[TlH]','[Pa]','[XeH]','[Os-3]','[PbH2]','[Fe+]','[Ir-3]','[as]','[Ra]','[PbH]','[Co-3]','[Fe-2]','[p+]','[Mo+]','[TlH2]','[No]','[pH+]','[Te+]','[Hg-]','[Sn-]','[Am]','[Al-2]','[Pm]','[SiH-]','[asH]','[In-]','[Ga-]','[IH+3]','[Ir-2]','[Ta-]','[FeH]','[IH2+3]','[Mn-2]','[Fe-]','[IH3]','[BiH2]','[IH2]','[IH4]','[PH2-]','[AsH4]','[WH]','[BH3-]','[At]','[Au]','[Ir+3]','[S@]','[Ru+]','[U+2]','[Al-]','[Cl-]','[Gd]','[Mn+]','[Mo]','[Cu]','[CH]','[P-]','[Fe]','[131I]','[I-]','[Mg+2]','[NH+]','[Ag+]','[Mn+2]','[Co]','[BH2-]','[n+]','[K]','[cH+]','[Ba+2]','[Sn+2]','[N@@]','[Sn+]','[Sb+]','[Ni+]','[123I]','[S@@]','[Co+2]','[pH]','[I+3]','[OH2+]','[I+2]','[V]','[B-]','[SnH]','[SnH2]','[Pt]','[S+2]','[La]','[Sb]','[W+]','[UH]','[AlH]','[SH+]','[PH4]','[Bi+]','[SeH]','[Fe-4]','[PH2]','[se+]','[SiH3]','[S-2]','[Cd+2]','[Ru-2]','[Mg]','[Si-]','[Yb]','[PH]','[Zr-2]','[U]','[Be+2]','[C@@]','[Fe+3]','[Ti+]','[AsH]','[Hg]','[SiH2]','[Cl+]','[SH]','[S]','[nH]','[BH-]','[Hf]','[Mn+3]','[Cu-2]','[Tb]','[N@@H+]','[OH-]','[P]','[PH-]','[H-]','[Tl]','[Zn+2]','[C@H]','[K+]','[SH2]','[Cu+2]','[IH]','[CH2-]','[PH+]','[Dy]','[O]','[N@+]','[GeH3]','[Li]','[Fe+2]','[O-2]','[99Tc]','[Ru]','[2H]','[18F]','[Cu-]','[S+]','[Ho]','[Ir]','[Mn]','[NH]','[Na+]','[Hf+2]','[MgH]','[Se+]','[Cr+2]','[C+]','[Se]','[Bi+2]','[ClH+]','[SH2+]','[N@]','[Pd]','[Sb-]','[In]','[Nd]','[W]','[OH+]','[o+]','[Ce]','[Eu]','[Cr]','[Pd-2]','[Al+2]','[Br+2]','[Rh]','[se]','[NH-]','[KH]','[BH]','[CH-]','[Bi]','[Rb]','[Sm]','[Se-]','[Rh+]','[N-]','[c-]','[cH-]','[AlH-]','[CaH2]','[Ni]','[Pb]','[Ac]','[NH2]','[C@@H]','[YH]','[NH2-]','[Sr+2]','[Pd-]','[VH]','[Cs+]','[Ir+]','[TeH]','[B+]','[N@@+]','[Fe-3]','[32P]','[La+3]','[C@]','[B+2]','[GeH]','[Te]','[AsH2]','[AlH2]','[Hg+2]','[Gd+3]','[Ti+2]','[P+]','[O-]','[GeH2]','[Ta]','[Ga]','[TeH2]','[s+]','[Pd+2]','[LiH]','[Si]','[Li+]','[SnH3]','[PH2+]','[Ca+2]','[Ge]','[Au+]','[Ag]','[CH2+]','[Sn+3]','[Zn]','[As]','[Os]','[Ga+3]','[Zr]','[Cu+]','[BrH+]','[SH-]','[NH3+]','[SbH]','[Al]','[Os+2]','[Tc]','[Rh+2]','[RuH2]','[Au-]','[Pt+2]','[Sn]','[Pt-2]','[NH2+]','[Na]','[Xe]','[n-]','[Hg+]','[S-]','[As+]','[CH+]','[NaH]','[Ti]','[P@@]','[Cr+3]','[N@H+]','[NH4+]','[AlH3]','[As+3]','[c+]','[CH2]','[C]','[N]','[Si+]','[Th]','[Al+]','[Zr+2]','[Cl]','[Pt-]','[Pb+2]','[Cr+]','[Cs]','[te]','[H+]','[SiH]','[Co+]','[Br-]','[Re]','[nH+]','[N+]','[Cl+3]','[Nb]','[Cl+2]','[Br+]','[Ba]','[H]','[Ni-]','[C-]','[Ni+2]','[Ru+2]','[O+]','[Y]','[Ca]','[SbH2]','[Cd]','[Ag-]','[I+]','[ClH2+]','[Mg+]','[Mo+2]','[IH2+]','[Pr]','[F-]','[B+3]','[Sc]','[Br]','[Tl+]','[Sb+3]'])    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)

    atom_hidden = 115
    bond_hidden = 13
    train_loader_mask = construct_dataset_mask(mol, atom_hidden=atom_hidden, bond_hidden=bond_hidden,
                                               max_length=max_length, max_ids=512, tokenizer=tokenizer)
