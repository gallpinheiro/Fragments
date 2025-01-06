# This code enables parallel execution, as outlined in the original work.
# Please, check the time complexity discussed in the paper (page 4).
# Due to the parallelization, the code contains additional lines, which may make the files harder to read.


# Motif-vocabulary Construction Phase (Phase 2) - see page 4

import multiprocessing as mp
import os
import os.path as path
import pickle
from collections import Counter
from datetime import datetime
from functools import partial
from typing import List, Tuple

from tqdm import tqdm

from mol_graph import MolGraph as MG

def apply_operations(batch):
    vocab = Counter()
    for smi in batch:
        mol = MG(smi, tokenizer='motif') # maybe k is defined here
        vocab = vocab + Counter(mol.motifs)
    return vocab

if __name__ == "__main__":
    
    mp.set_start_method('fork')
    num_workers = 2
    data_set = ['CCCCC', 'CCC', 'CCCCCCCCCCCCC', 'CCCCOC', 'CC', 'C1cccccc1C']
    batch_size = (len(data_set) - 1) // num_workers + 1
    batches = [data_set[i : i + batch_size] for i in range(0, len(data_set), batch_size)]
    print(f'Total: {len(data_set)} molecules.\n')
    num_operations = 500 # Determine how many operations to recover from phase 1.

    print(f'Processing...')
    vocab = Counter()
    # Loads all fragments founded in phase 1
    MG.load_operations('merging_operation.txt', num_operations)
    # Convert SMILES strings to graph (MG obj see mol_graph.py)
    # 'vocab = Counter()' is a dictionary that stores the frequency of each fragment in the dataset's molecules.
    # The key represents the fragment, and the value represents its frequency.
    with mp.Pool(num_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as pool:
        for batch_vocab in pool.imap(apply_operations, batches):
            vocab = vocab + batch_vocab

    # 'MG.OPERATIONS' contains the merging operations performed during phase 1.
    atom_list = [x for (x, _) in vocab.keys() if x not in MG.OPERATIONS]
    atom_list.sort()
    new_vocab = []
    full_list = atom_list + MG.OPERATIONS
    for (x, y), value in vocab.items():
        assert x in full_list
        new_vocab.append((x, y, value))
        
    index_dict = dict(zip(full_list, range(len(full_list))))
    sorted_vocab = sorted(new_vocab, key=lambda x: index_dict[x[0]])
    with open('vocab.txt', 'w') as f:
        for (x, y, _) in sorted_vocab:
            f.write(f"{x} {y}\n")
    
    print(f"\r[{datetime.now()}] Motif vocabulary construction finished.")
