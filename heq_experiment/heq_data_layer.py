import numpy as np
import torch
 
from heq_causal_model import compute_states, counterfactual_labels
from constants import EMBEDDING_DIM, INPUT_VARS, NUM_ENTITIES
 
ENTITY_SEED = 0   # the table is fixed; never tie this to the run seed
 
 
def build_entity_vectors(num_entities=NUM_ENTITIES, embedding_dim=EMBEDDING_DIM, seed=ENTITY_SEED):
    """Fixed random entity vectors in [-1, 1]^d. [num_entities, embedding_dim]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(num_entities, embedding_dim)).astype(np.float32)
 
 
def sample_rows(size, seed, num_entities=NUM_ENTITIES, balanced=False):
    """[size, 4] entity-id rows (W,X,Y,Z), WITH replacement.
    balanced=True forces each pair's equality 50/50 (kept for the forward dataset)."""
    rng = np.random.default_rng(seed)
    if not balanced:
        return rng.integers(0, num_entities, size=(size, len(INPUT_VARS)), dtype=np.int64)
    cols = []
    for _ in range(len(INPUT_VARS) // 2):
        equal = rng.integers(0, 2, size=size).astype(bool)
        a = rng.integers(0, num_entities, size=size)
        b = np.where(equal, a, (a + rng.integers(1, num_entities, size=size)) % num_entities)
        cols += [a, b]
    return np.stack(cols, axis=1)
 
 
def sample_unique_rows(size, seed, num_entities=NUM_ENTITIES):
    """[size, 4] DISTINCT rows drawn uniformly without replacement from the
    num_entities**4 grid (repo-style). Skewed naturally: P(W==X) = 1/num_entities."""
    total = num_entities ** len(INPUT_VARS)
    if size > total:
        raise ValueError(f"size={size} exceeds {total} unique four-entity rows")
    rng = np.random.default_rng(seed)
    values = rng.choice(total, size=size, replace=False)
    rows = np.zeros((size, len(INPUT_VARS)), dtype=np.int64)
    for col in range(len(INPUT_VARS) - 1, -1, -1):
        rows[:, col] = values % num_entities
        values //= num_entities
    return rows
 
 
def rows_to_inputs(rows, entity_vectors):
    """[N, 4] ids -> [N, 4 * embedding_dim] float32 tensor (concatenated vectors)."""
    rows = np.atleast_2d(np.asarray(rows, dtype=np.int64))
    return torch.tensor(entity_vectors[rows].reshape(rows.shape[0], -1), dtype=torch.float32)
 
 
def make_dataset(entity_vectors, size, seed):
    """Forward task: (X [N,16], y [N]=O). Uniform rows -> natural ~0.91 skew."""
    rows = sample_rows(size, seed)
    return rows_to_inputs(rows, entity_vectors), torch.tensor(compute_states(rows)["O"], dtype=torch.long)
 
 
# --- pair-bank policy helpers (mirror the repo) ----------------------------- #
 
def _policy_positive_mask(base_O, cf_by_var, target="any"):
    """Which pairs are 'positive' = the interchange changes O.
    target 'any' -> any var's swap flips O; 'WX'/'YZ' -> that var; 'both' -> both."""
    changed = {var: (cf_by_var[var] != base_O) for var in cf_by_var}
    if target == "any":
        return np.logical_or.reduce(list(changed.values()))
    if target in changed:
        return changed[target]
    if target == "both":
        return np.logical_and.reduce(list(changed.values()))
    raise ValueError(f"unknown pair_policy_target={target}")
 
 
def _select_pair_indices(positive_mask, size, mixed_positive_fraction, rng):
    """Repo-style mixed selection: take a `mixed_positive_fraction` share of positive
    pairs (interchange changes O) and the rest negative. Returns indices [size]."""
    n = int(positive_mask.shape[0])
    if size > n:
        raise ValueError(f"size={size} exceeds pool={n}; raise pool_factor")
    shuffled = rng.permutation(n)
    pos_target = int(np.floor(size * float(mixed_positive_fraction) + 0.5))
    pos_target = min(max(pos_target, 0), size)
    neg_target = size - pos_target
    selected, pos, neg = [], 0, 0
    for i in shuffled:
        if positive_mask[i]:
            if pos >= pos_target:
                continue
            pos += 1
        else:
            if neg >= neg_target:
                continue
            neg += 1
        selected.append(int(i))
        if len(selected) == size:
            return np.asarray(selected, dtype=np.int64)
    raise ValueError(f"could not build {size} mixed pairs; raise pool_factor")
 
 
def _sample_pair_rows(size, seed, pair_policy, pair_policy_target,
                      mixed_positive_fraction, pool_factor):
    """Sample (base_rows, source_rows) under the given policy. unfiltered -> skewed
    as-is; mixed -> oversample a pool then select to hit the positive fraction."""
    if pair_policy == "unfiltered":
        return sample_unique_rows(size, seed), sample_unique_rows(size, seed + 1)
    if pair_policy != "mixed":
        raise ValueError(f"unknown pair_policy={pair_policy}")
 
    pool = size * pool_factor
    base = sample_unique_rows(pool, seed)
    source = sample_unique_rows(pool, seed + 1)
    cf = counterfactual_labels(compute_states(base), compute_states(source))
    mask = _policy_positive_mask(compute_states(base)["O"], cf, pair_policy_target)
    idx = _select_pair_indices(mask, size, mixed_positive_fraction, np.random.default_rng(seed + 2))
    return base[idx], source[idx]
 
 
def make_data_pair(var, entity_vectors, size, seed, pair_policy="unfiltered",
                   mixed_positive_fraction=0.5, pool_factor=8):
    """Interchange task for one variable: (base [N,16], source [N,16], targets [N], base_labels [N]).
    Repo default pair_policy='unfiltered' (skewed); 'mixed' balances on whether the
    swap of `var` changes O."""
    base, source = _sample_pair_rows(size, seed, pair_policy, var, mixed_positive_fraction, pool_factor)
    base_states, source_states = compute_states(base), compute_states(source)
    targets = counterfactual_labels(base_states, source_states)[var]
    return (rows_to_inputs(base, entity_vectors),
            rows_to_inputs(source, entity_vectors),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(base_states["O"], dtype=torch.long))   # base_labels
 
 
def make_plot_bank(entity_vectors, size, seed, pair_policy="unfiltered",
                   pair_policy_target="any", mixed_positive_fraction=0.5, pool_factor=8):
    """One shared interchange bank for PLOT: base.O and counterfactuals for ALL variables.
 
    Returns (base_inputs [N,16], source_inputs [N,16], base_labels [N], cf_by_var {var:[N]}).
    Repo default pair_policy='unfiltered' (skewed, no rebalance); 'mixed' balances pairs
    on causal effect (changed_any) via pair_policy_target.
    """
    base, source = _sample_pair_rows(size, seed, pair_policy, pair_policy_target,
                                     mixed_positive_fraction, pool_factor)
    base_states, source_states = compute_states(base), compute_states(source)
    return (rows_to_inputs(base, entity_vectors),
            rows_to_inputs(source, entity_vectors),
            base_states["O"],
            counterfactual_labels(base_states, source_states))