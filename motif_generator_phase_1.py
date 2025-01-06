# This code enables parallel execution, as outlined in the original work.
# Please, check the time complexity discussed in the paper (page 4).
# Due to the parallelization, the code contains additional lines, which may make the files harder to read.


# Merging-operation learning phase (Phase 1) - see page 3 

import multiprocessing as mp
from datetime import datetime

from utils import *


if __name__ == "__main__":
    mp.set_start_method('fork')

    # EXAMPLE
    smiles_list = ['CCCCC', 'CCC', 'CCCCCCCCCCCCC', 'CCCCOC', 'CC', 'C1cccccc1C']

    num_workers = 3
    num_iters = 3000
    min_frequency = 0
    mp_threshold = 1e5
    
    smiles_list = [(i, smi) for (i, smi) in enumerate(smiles_list)]
    batch_size = (len(smiles_list) - 1) // num_workers + 1
    batches = [smiles_list[i : i + batch_size] for i in range(0, len(smiles_list), batch_size)]
    mols = []
    # Convert SMILES strings to graph (MolGraph obj see load_batch_mols in utils.py)
    with mp.Pool(num_workers) as pool:
        for mols_batch in pool.imap(load_batch_mols, batches):
            mols.extend(mols_batch)

    # Check if a pair of nodes forms a fragment that belongs to the molecule (using the adjacency matrix).
    # This step identifies the initial fragments.
    # 'stats' keeps track of the total count for each fragment across all molecules.
    # 'indices' records how many of each fragment are present in each molecule.
    stats, indices = get_stats(mols, num_workers)

    output = open('merging_operation.txt', 'w')
    for i in range(num_iters): # K iteration (see page 3 on paper)
        print(f'[{datetime.now()}] Iteration {i}.')
        motif = max(stats, key=lambda x: (stats[x], x)) # Retrieve the fragment with the highest number of occurrences.
        if stats[motif] < min_frequency:
            # If the fragment with the highest occurrence count is less than the threshold, no action is taken.
            print(f'No motif has frequency >= {min_frequency}. Stopping.\n')
            break
        print(f'[Iteration {i}] Most frequent motif: {motif}, frequency: {stats[motif]}.\n')

        # Magic happens here:
        # First, it identifies the molecules containing the most popular fragment.
        # It uses the 'indices' variable to retrieve this information.
        # Next, it merges fragments into new ones and updates the 'stats' and 'indices' variables accordingly.
        # The statistics for the most popular fragments are reset to zero to avoid reprocessing them.
        apply_merging_operation(
            motif=motif,
            mols=mols,
            stats=stats,
            indices=indices,
            num_workers=num_workers if stats[motif] >= mp_threshold else 1,
        )
        # Save the most popular fragment to a file. 
        output.write(f'{motif}\n')

    # Close the file.
    output.close()