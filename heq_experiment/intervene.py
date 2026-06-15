import numpy as np
import torch
import torch.nn.functional as F
from ot_solver import solve_ot, solve_uot
 
 
def variable_signature(factual_output, list_counterfactual_output, num_classes=NUM_CLASSES):
    base_onehot = F.one_hot(torch.as_tensor(factual_output, dtype=torch.long), num_classes).float()
    var_signatures, names = [], []
    for var_changed, cf_output in list_counterfactual_output.items():
        cf_onehot = F.one_hot(torch.as_tensor(cf_output, dtype=torch.long), num_classes).float()
        var_signatures.append((cf_onehot - base_onehot).reshape(-1))
        names.append(var_changed)
    return torch.stack(var_signatures), names
 
 
 
def layer_directions(activations, mode, k):
    if mode == "neuron":
        return torch.eye(activations.shape[-1], dtype=activations.dtype)
    if mode == "pca":
        centered = activations - activations.mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        k = Vh.shape[0] if k is None else min(k, Vh.shape[0])
        return Vh[:k].T
 
 
@torch.no_grad()
def site_signature(model, base, source, mode="neuron", k=None, num_classes=NUM_CLASSES):
    model.eval()
 
    def retrieve_list_activations(x):
        h = x.unsqueeze(1)   # (N, 1, W)
        list_activations = []
        for block in model.blocks:
            h = block(h)
            list_activations.append(h)
        return list_activations
 
    base_acts = retrieve_list_activations(base)
    source_acts = retrieve_list_activations(source)
    base_prob = F.softmax(model.score(base_acts[-1]).squeeze(1), dim=-1)   # [N, C]
 
    num_ex = base.shape[0]
    rows, sites, directions = [], [], {}
    for layer_id in range(len(model.blocks)):
        current_base_act = base_acts[layer_id].squeeze(1)        # [N, W]
        current_source_act = source_acts[layer_id].squeeze(1)
        width = current_base_act.shape[-1]
        # each col of V is a projection vector
        V = layer_directions(current_base_act, mode, k) # [W, D]
        directions[layer_id] = V
        num_dirs = V.shape[1]
 
        coeff = (current_source_act - current_base_act) @ V # [N, D]
        intervened_acts = current_base_act.unsqueeze(0) + coeff.T[:, :, None] * V.T[:, None, :]  # [D, N, W]
        intervened_acts = intervened_acts.reshape(num_dirs * num_ex, 1, width)
        for block in model.blocks[layer_id + 1:]:
            intervened_acts = block(intervened_acts)
 
        prob = F.softmax(model.score(intervened_acts).squeeze(1), dim=-1).reshape(num_dirs, num_ex, num_classes)
        signature = prob - base_prob.unsqueeze(0)                # [D, N, C]
        for d in range(num_dirs):
            rows.append(signature[d].reshape(-1))
            sites.append((layer_id, d))
 
    return torch.stack(rows), sites, directions
 
 
def evaluate_handle(model, base_inputs, source_inputs, cf_targets, T, causal_var_idx, sites, directions,
                    top_k, soft=False):
                    
    score_sites = T[causal_var_idx]
    ordered_sites = score_sites.argsort(descending=True)[:top_k].tolist()

    norm = max(float(score_sites[j]) for j in ordered_sites) if soft else 0.0
 
    columns = {}
    for j in ordered_sites:

        layer_id, direction_id = sites[j]
        v = directions[layer_id][:, direction_id]

        weight = (score_sites[j] / norm) if soft else 1.0

        if layer_id not in columns:
            columns[layer_id] = []
        
        columns[layer_id].append(v * (weight ** 0.5))
 
    list_places_to_intervene = [None] * len(model.blocks)
    for layer, list_directions in columns.items():
        layer_directions = torch.stack(list_directions, dim=1)
        list_places_to_intervene[layer] = layer_directions @ layer_directions.transpose(-1, -2)
 
    intervened_logits = intervene(model, base_inputs, source_inputs, list_places_to_intervene, strength=1.0)
    return (intervened_logits.argmax(-1) == cf_targets).float().mean().item()

 
def run_plot_no_autotune(model, entity_vectors, mode="neuron", k=None, calib_size=4000, test_size=2000,
             eps=0.05, tau=1.0, reg_m=(1.0, 1.0), top_k=1, soft=False, seed=0):
    cb, cs, c_base_O, c_cf = make_plot_bank(entity_vectors, calib_size, seed=seed)
    G, var_names = variable_signature(c_base_O, c_cf)
    S, sites, directions = site_signature(model, cb, cs, mode=mode, k=k)
    T = solve_uot(G, S, eps=eps, tau=tau, reg_m=reg_m)
 
    hb, hs, h_base_O, h_cf = make_plot_bank(entity_vectors, test_size, seed=seed + 100)
    out = {}
    for i, var in enumerate(var_names):
        targets = torch.as_tensor(h_cf[var], dtype=torch.long)
        out[var] = evaluate_handle(model, hb, hs, targets, T, i, sites, directions,
                                   top_k=top_k, soft=soft)
    return out


def run_plot(model, entity_vectors, mode="neuron", k=None,
             train_size=128, calib_size=128, test_size=512,
             eps=0.05, tau=1.0, reg_m=(1.0, 1.0),
             top_k_values=[i for i in range(1, 20)], soft=False, seed=0):
    # obtain T
    tb, ts, t_base_O, t_cf = make_plot_bank(entity_vectors, train_size, seed=seed)
    G, var_names = variable_signature(t_base_O, t_cf)
    S, sites, directions = site_signature(model, tb, ts, mode=mode, k=k)
    T = solve_ot(G, S, eps=eps, tau=tau)

    # with fixed T, tune the hyperparameter
    cb, cs, c_base_O, c_cf = make_plot_bank(entity_vectors, calib_size, seed=seed + 50)
    best_top_k = {}
    for i, var in enumerate(var_names):
        c_targets = torch.as_tensor(c_cf[var], dtype=torch.long)
        scores = {tk: evaluate_handle(model, cb, cs, c_targets, T, i, sites, directions,
                                      top_k=tk, soft=soft) for tk in top_k_values}
        best_top_k[var] = max(scores, key=scores.get)
        # print(var, scores)

    # 3) test with the best configuration
    hb, hs, h_base_O, h_cf = make_plot_bank(entity_vectors, test_size, seed=seed + 100)
    out = {}
    for i, var in enumerate(var_names):
        h_targets = torch.as_tensor(h_cf[var], dtype=torch.long)
        out[var] = evaluate_handle(model, hb, hs, h_targets, T, i, sites, directions,
                                   top_k=best_top_k[var], soft=soft)
    return out, best_top_k