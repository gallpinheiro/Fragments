from collections import defaultdict
from multiprocessing import Process, Queue
from typing import Dict, List, Tuple

import networkx as nx
import rdkit.Chem as Chem
from rdkit import Chem
from rdkit.Chem import AllChem


class MolGraph:

    def __init__(self, smiles, idx=0):
        self.idx = idx
        self.mol_graph = Chem.MolFromSmiles(smiles)
        self.merging_graph = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol_graph))
        for atom in self.mol_graph.GetAtoms():
            self.merging_graph.nodes[atom.GetIdx()]['atom_indices'] = set([atom.GetIdx()])
    
    def apply_merging_operation(self, motif, stats, indices):
        if self.merging_graph.number_of_nodes() == 1:
            return
        new_graph = self.merging_graph.copy()
        for (node1, node2) in self.merging_graph.edges:
            if not new_graph.has_edge(node1, node2):
                continue
            atom_indices = new_graph.nodes[node1]['atom_indices'].union(new_graph.nodes[node2]['atom_indices'])
            motif_smiles = fragment2smiles(self, atom_indices)
            if motif_smiles == motif:
                graph_before_merge = new_graph.copy()
                merge_nodes(new_graph, node1, node2)
                update_stats(self, graph_before_merge, new_graph, node1, node2, stats, indices, self.idx)
        self.merging_graph = new_graph
        indices[motif][self.idx] = 0
    
    def apply_merging_operation_producer(self, motif: str, q) -> None:
        if self.merging_graph.number_of_nodes() == 1:
            return
        new_graph = self.merging_graph.copy()
        for (node1, node2) in self.merging_graph.edges:
            if not new_graph.has_edge(node1, node2):
                continue
            atom_indices = new_graph.nodes[node1]['atom_indices'].union(new_graph.nodes[node2]['atom_indices'])
            motif_smiles = fragment2smiles(self, atom_indices)
            if motif_smiles == motif:
                graph_before_merge = new_graph.copy()
                merge_nodes(new_graph, node1, node2)
                update_stats_producer(self, graph_before_merge, new_graph, node1, node2, q, self.idx)
        q.put((motif, self.idx, new_graph))


def merge_nodes(graph: nx.Graph, node1: int, node2: int) -> None:
    neighbors = [n for n in graph.neighbors(node2)]
    atom_indices = graph.nodes[node1]["atom_indices"].union(graph.nodes[node2]["atom_indices"])
    for n in neighbors:
        if node1 != n and not graph.has_edge(node1, n):
            graph.add_edge(node1, n)
        graph.remove_edge(node2, n)
    graph.remove_node(node2)
    graph.nodes[node1]["atom_indices"] = atom_indices


def fragment2smiles(mol: Chem.rdchem.Mol, indices: List[int]) -> str:
    smiles = Chem.MolFragmentToSmiles(mol.mol_graph, tuple(indices))
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=False))


def get_conn_list(motif: Chem.rdchem.Mol, use_Isotope: bool=False, symm: bool=False) -> Tuple[List[int], Dict[int, int]]:

    ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False, breakTies=False))
    if use_Isotope:
        ordermap = {atom.GetIsotope(): ranks[atom.GetIdx()] for atom in motif.GetAtoms() if atom.GetSymbol() == '*'}
    else:
        ordermap = {atom.GetIdx(): ranks[atom.GetIdx()] for atom in motif.GetAtoms() if atom.GetSymbol() == '*'}
    if len(ordermap) == 0:
        return [], {}
    ordermap = dict(sorted(ordermap.items(), key=lambda x: x[1]))
    if not symm:
        conn_atoms = list(ordermap.keys())
    else:
        cur_order, conn_atoms = -1, []
        for idx, order in ordermap.items():
            if order != cur_order:
                cur_order = order
                conn_atoms.append(idx)
    return conn_atoms, ordermap


def graph2smiles(fragment_graph: nx.Graph, with_idx: bool=False) -> str:
    motif = Chem.RWMol()
    node2idx = {}
    for node in fragment_graph.nodes:
        idx = motif.AddAtom(smarts2atom(fragment_graph.nodes[node]['smarts']))
        if with_idx and fragment_graph.nodes[node]['smarts'] == '*':
            motif.GetAtomWithIdx(idx).SetIsotope(node)
        node2idx[node] = idx
    for node1, node2 in fragment_graph.edges:
        motif.AddBond(node2idx[node1], node2idx[node2], fragment_graph[node1][node2]['bondtype'])
    return Chem.MolToSmiles(motif, allBondsExplicit=True)


def smiles2mol(smiles: str, sanitize: bool=False) -> Chem.rdchem.Mol:
    if sanitize:
        return Chem.MolFromSmiles(smiles)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    AllChem.SanitizeMol(mol, sanitizeOps=0)
    return mol


def smarts2atom(smarts: str) -> Chem.rdchem.Atom:
    return Chem.MolFromSmarts(smarts).GetAtomWithIdx(0)


def get_stats_producer(batch, q):
    for mol in batch:
        for (node1, node2) in mol.merging_graph.edges:
            atom_indices = mol.merging_graph.nodes[node1]['atom_indices'].union(mol.merging_graph.nodes[node2]['atom_indices'])
            motif_smiles = fragment2smiles(mol, atom_indices)
            q.put((mol.idx, motif_smiles))
    q.put(None)


def get_stats_consumer(stats, indices, q, num_workers):
    num_tasks_done = 0
    while True:
        info = q.get()
        if info == None:
            num_tasks_done += 1
            if num_tasks_done == num_workers:
                break
        else:
            (idx, smi) = info
            stats[smi] += 1
            indices[smi][idx] += 1


def get_stats(mols, num_workers):

    stats = defaultdict(int)
    indices = defaultdict(lambda: defaultdict(int))
    
    if num_workers == 1:
        for mol in mols:
            for (node1, node2) in mol.merging_graph.edges:
                atom_indices = mol.merging_graph.nodes[node1]['atom_indices'].union(mol.merging_graph.nodes[node2]['atom_indices'])
                motif_smiles = fragment2smiles(mol, atom_indices)
                stats[motif_smiles] += 1
                indices[motif_smiles][mol.idx] += 1
    else:
        batch_size = (len(mols) - 1) // num_workers + 1
        batches = [mols[i : i + batch_size] for i in range(0, len(mols), batch_size)]
        q = Queue()
        producers = [Process(target=get_stats_producer, args=(batches[i], q)) for i in range(num_workers)]
        [p.start() for p in producers]
        get_stats_consumer(stats, indices, q, num_workers)
        [p.join() for p in producers]
    return stats, indices


def load_batch_mols(batch):
    return [MolGraph(smi, idx) for (idx, smi) in batch]


def apply_merging_operation(
    motif, mols, stats, indices, num_workers=1
):
    mols_to_process = [mols[i] for i, freq in indices[motif].items() if freq > 0]
    if num_workers > 1:
        batch_size = (len(mols_to_process) -1 ) // num_workers + 1
        batches = [mols_to_process[i : i + batch_size] for i in range(0, len(mols_to_process), batch_size)]
        q = Queue()
        producers = [Process(target=apply_merging_operation_producer, args=(motif, batches[i], q)) for i in range(num_workers)]
        [p.start() for p in producers]
        apply_merging_operation_consumer(mols, stats, indices, q, num_workers)
        [p.join() for p in producers]
    else:
        [mol.apply_merging_operation(motif, stats, indices) for mol in mols_to_process]
    stats[motif] = 0


def apply_merging_operation_producer(motif, batch, q):
    [mol.apply_merging_operation_producer(motif, q) for mol in batch]
    q.put(None)

def apply_merging_operation_consumer(mols, stats, indices, q, num_workers):
    num_tasks_done = 0
    while True:
        info = q.get()
        if info == None:
            num_tasks_done += 1
            if num_tasks_done == num_workers:
                break
        else:
            (motif, i, change) = info
            if isinstance(change, int):
                stats[motif] += change
                indices[motif][i] += change
            else:
                assert isinstance(change, nx.Graph)
                indices[motif][i] = 0
                mols[i].merging_graph = change


def update_stats(mol, graph, new_graph, node1, node2, stats, indices, i):
    neighbors1 = [n for n in graph.neighbors(node1)]
    for n in neighbors1:
        if n != node2:
            atom_indices = graph.nodes[node1]["atom_indices"].union(graph.nodes[n]["atom_indices"])
            motif_smiles = fragment2smiles(mol, atom_indices)
            stats[motif_smiles] -= 1
            indices[motif_smiles][i] -= 1
    neighbors2 = [n for n in graph.neighbors(node2)]
    for n in neighbors2:
        if n != node1:
            atom_indices = graph.nodes[node2]["atom_indices"].union(graph.nodes[n]["atom_indices"])
            motif_smiles = fragment2smiles(mol, atom_indices)
            stats[motif_smiles] -= 1
            indices[motif_smiles][i] -= 1
    neighbors = [n for n in new_graph.neighbors(node1)]
    for n in neighbors:
        atom_indices = new_graph.nodes[node1]["atom_indices"].union(new_graph.nodes[n]["atom_indices"])
        motif_smiles = fragment2smiles(mol, atom_indices)
        stats[motif_smiles] += 1
        indices[motif_smiles][i] += 1

def update_stats_producer(mol, graph, new_graph, node1, node2, q, i):
    neighbors1 = [n for n in graph.neighbors(node1)]
    for n in neighbors1:
        if n != node2:
            atom_indices = graph.nodes[node1]["atom_indices"].union(graph.nodes[n]["atom_indices"])
            motif_smiles = fragment2smiles(mol, atom_indices)
            q.put((motif_smiles, i, -1))
    neighbors2 = [n for n in graph.neighbors(node2)]
    for n in neighbors2:
        if n != node1:
            atom_indices = graph.nodes[node2]["atom_indices"].union(graph.nodes[n]["atom_indices"])
            motif_smiles = fragment2smiles(mol, atom_indices)
            q.put((motif_smiles, i, -1))
    neighbors = [n for n in new_graph.neighbors(node1)]
    for n in neighbors:
        atom_indices = new_graph.nodes[node1]["atom_indices"].union(new_graph.nodes[n]["atom_indices"])
        motif_smiles = fragment2smiles(mol, atom_indices)
        q.put((motif_smiles, i, 1))


class MotifVocab(object):

    def __init__(self, pair_list: List[Tuple[str, str]]):
        self.motif_smiles_list = [motif for _, motif in pair_list]
        self.motif_vmap = dict(zip(self.motif_smiles_list, range(len(self.motif_smiles_list))))

        node_offset, conn_offset, num_atoms_dict, nodes_idx = 0, 0, {}, []
        vocab_conn_dict: Dict[int, Dict[int, int]] = {}
        conn_dict: Dict[int, Tuple[int, int]] = {}
        bond_type_motifs_dict = defaultdict(list)
        for motif_idx, motif_smiles in enumerate(self.motif_smiles_list):
            motif = smiles2mol(motif_smiles)
            ranks = list(Chem.CanonicalRankAtoms(motif, includeIsotopes=False, breakTies=False))

            cur_orders = []
            vocab_conn_dict[motif_idx] = {}
            for atom in motif.GetAtoms():
                if atom.GetSymbol() == '*' and ranks[atom.GetIdx()] not in cur_orders:
                    bond_type = atom.GetBonds()[0].GetBondType()
                    vocab_conn_dict[motif_idx][ranks[atom.GetIdx()]] = conn_offset
                    conn_dict[conn_offset] = (motif_idx, ranks[atom.GetIdx()])
                    cur_orders.append(ranks[atom.GetIdx()])
                    bond_type_motifs_dict[bond_type].append(conn_offset)
                    nodes_idx.append(node_offset)
                    conn_offset += 1
                node_offset += 1
            num_atoms_dict[motif_idx] = motif.GetNumAtoms()
        self.vocab_conn_dict = vocab_conn_dict
        self.conn_dict = conn_dict
        self.nodes_idx = nodes_idx
        self.num_atoms_dict = num_atoms_dict
        self.bond_type_conns_dict = bond_type_motifs_dict


    def __getitem__(self, smiles: str) -> int:
        if smiles not in self.motif_vmap:
            print(f"{smiles} is <UNK>")
        return self.motif_vmap[smiles] if smiles in self.motif_vmap else -1
    
    def get_conn_label(self, motif_idx: int, order_idx: int) -> int:
        return self.vocab_conn_dict[motif_idx][order_idx]
    
    def get_conns_idx(self) -> List[int]:
        return self.nodes_idx
    
    def from_conn_idx(self, conn_idx: int) -> Tuple[int, int]:
        return self.conn_dict[conn_idx]


class Vocab(object):
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vmap = dict(zip(self.vocab_list, range(len(self.vocab_list))))
        
    def __getitem__(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab_list[idx]

    def size(self):
        return len(self.vocab_list)