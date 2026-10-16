"""Protocol and completed-update token accounting for the batch/concurrency matrix."""
import random


def matrix_plan(batches=(4, 8), levels=(1, 2, 4, 8), repeats=3, seed=42):
    if (not batches or not levels or len(set(batches)) != len(batches) or
        len(set(levels)) != len(levels) or min(batches) < 1 or min(levels) < 1 or repeats < 1):
        raise ValueError('Unique positive batch sizes/levels and positive repeats required')
    cells = [(b, n) for b in batches for n in levels]
    cells.extend(cell for cell in [(1, 1), (1, max(levels))] if cell not in cells)
    cells = list(dict.fromkeys(cells))
    rng = random.Random(seed)
    plan = []
    for repeat in range(1, repeats + 1):
        block = list(cells)
        rng.shuffle(block)
        plan.extend({'batch_size': batch, 'n': n, 'repeat': repeat} for batch, n in block)
    return plan


def update_token_counts(sample_counts, batch_size):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Positive integer physical batch required')
    return {name: [sum(counts[i:i+batch_size]) for i in range(0, len(counts), batch_size)]
            for name, counts in sample_counts.items()}
