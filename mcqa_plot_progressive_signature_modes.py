import os
import string

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from mcqa_data_load_all import build_mcqa_banks, letter_token_id

from mcqa_neural_net import load_gemma_model
from mcqa_ot import solve_ot, solve_uot
from mcqa_utils import set_seed, normalize_rows


FAMILY_ORDER = ("answer_pointer", "answer_token", "both")
TARGETS = ("answer_pointer", "answer_token")
ANSWER_LETTERS = tuple(string.ascii_uppercase)


def answer_label_ids(tokenizer):
    """Return tokenizer ids for A-Z answer labels."""
    ids = []
    for letter in ANSWER_LETTERS:
        ids.append(letter_token_id(tokenizer, letter))
    return ids


def get_solver(name):
    """Return the OT or UOT solver selected by name."""
    if name == "ot":
        return solve_ot
    if name == "uot":
        return solve_uot
    raise ValueError(f"unknown solver={name!r}")


def make_signature(X, method="family_mean", pair_source_families=None, family_order=FAMILY_ORDER, eps=1e-8):
    """Build one signature from per-example features [N, D]."""
    X = torch.as_tensor(X, dtype=torch.float32)
    if X.ndim != 2:
        raise ValueError("X must have shape [N, D]")

    # if method == "concat":
    #     X = normalize_rows(X, eps)
    #     return X.reshape(-1)
    if method == "concat":
        signature = X.reshape(-1)
        signature = signature - signature.mean()
        return signature / signature.norm().clamp_min(eps)


    if method == "family_mean":
        if pair_source_families is None or len(pair_source_families) != X.shape[0]:
            raise ValueError("pair_source_families must have length N")

        blocks = []
        for family in family_order:
            mask_values = []
            for current_family in pair_source_families:
                mask_values.append(current_family == family)
            mask = torch.tensor(mask_values, dtype=torch.bool, device=X.device)
            block = X[mask].mean(dim=0) if bool(mask.any()) else torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
            block = block - block.mean()
            norm = torch.linalg.vector_norm(block)
            if float(norm.item()) > eps:
                block = block / norm
            blocks.append(block)

        return torch.cat(blocks, dim=0)

    raise ValueError(f"unknown signature method={method!r}")


def variable_signature(bank, num_labels=26, signature_method="family_mean", family_order=FAMILY_ORDER):
    """Build causal-variable signatures from label deltas on D_ft."""
    base = torch.as_tensor(bank["base_answer_label_ids"], dtype=torch.long)
    base_onehot = F.one_hot(base, num_classes=num_labels).float()
    signatures, names = [], []

    for name in TARGETS:
        cf = torch.as_tensor(bank["counterfactual_label_ids"][name], dtype=torch.long)
        delta = F.one_hot(cf, num_classes=num_labels).float() - base_onehot
        signatures.append(make_signature(delta, method=signature_method, pair_source_families=bank["pair_source_families"], family_order=family_order))
        names.append(name)

    return torch.stack(signatures, dim=0), names


def basis_dim(mode, n_rows, hidden_size, k=None, max_fit_states=4096):
    """Return the usable native/PCA dimension for site construction."""
    if mode == "neuron":
        return int(hidden_size)
    if mode != "pca":
        raise ValueError(f"unknown mode={mode!r}")
    n_fit = 2 * int(n_rows)
    if max_fit_states is not None:
        n_fit = min(n_fit, int(max_fit_states))
    if k is not None:
        n_fit = min(n_fit, int(k))
    return min(n_fit, int(hidden_size))


def fit_bases(sites, base_states, source_states, mode="neuron", k=None, max_fit_states=4096):
    """Fit one PCA basis per layer/token key, or mark keys for native-neuron patching."""
    keys = []
    for L, token_id, _, _ in sites:
        key = (int(L), token_id)
        if key not in keys:
            keys.append(key)

    bases = {}
    for key in keys:
        if mode == "neuron":
            bases[key] = {"mode": "neuron"}
            continue
        if mode != "pca":
            raise ValueError(f"unknown mode={mode!r}")

        X = torch.cat([base_states[key], source_states[key]], dim=0).float()
        if max_fit_states is not None and len(X) > int(max_fit_states):
            X = X[torch.randperm(len(X))[:int(max_fit_states)]]
        k_eff = min(len(X), X.shape[1])
        if k is not None:
            k_eff = min(k_eff, int(k))
        pca = PCA(n_components=k_eff, whiten=False).fit(X.numpy())
        bases[key] = {"mode": "pca", "components": torch.tensor(pca.components_, dtype=torch.float32)}
    return bases


def last_token_logits(model, outputs, attention_mask, label_ids):
    """Return raw logits over selected answer labels at the last real token."""
    device = next(model.parameters()).device
    rows = torch.arange(attention_mask.shape[0], device=device)
    cols = torch.arange(attention_mask.shape[1], device=device)
    pos = (attention_mask.to(device) * cols.unsqueeze(0)).max(dim=1).values
    hidden = outputs.last_hidden_state[rows, pos, :]
    ids = torch.tensor(label_ids, dtype=torch.long, device=device)
    logits = hidden @ model.lm_head.weight[ids].to(hidden.dtype).T

    bias = getattr(model.lm_head, "bias", None)
    if bias is not None:
        logits = logits + bias[ids]

    softcap = getattr(model.config, "final_logit_softcapping", None)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap

    return logits.float()


@torch.no_grad()
def collect_site_activations(model, input_ids, attention_mask, position_by_id, sites, batch_size=32, return_logits=False, label_ids=None):
    """Collect full hidden vectors at all layer/token keys needed by sites; optionally also return answer logits."""
    device = next(model.parameters()).device
    layer_ids, token_ids, keys = [], [], []
    states, logits = {}, []

    for L, token_id, _, _ in sites:
        L = int(L)
        key = (L, token_id)
        if L not in layer_ids:
            layer_ids.append(L)
        if token_id not in token_ids:
            token_ids.append(token_id)
        if key not in states:
            states[key] = []
            keys.append(key)

    layer_ids = sorted(layer_ids)
    N = input_ids.shape[0]

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        rows = torch.arange(ids.shape[0], device=device)
        pad_offset = (mask == 0).sum(dim=1)
        pos_by_token = {}

        for token_id in token_ids:
            raw_pos = position_by_id[token_id][start:end].to(device)
            pos_by_token[token_id] = pad_offset + raw_pos

        handles = []

        def make_hook(layer_id):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                for key in keys:
                    L_key, token_id = key
                    if L_key == layer_id:
                        pos = pos_by_token[token_id]
                        states[key].append(hidden[rows, pos, :].detach().float().cpu())
            return hook

        for L in layer_ids:
            handles.append(model.model.layers[L].register_forward_hook(make_hook(L)))

        try:
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
            if return_logits:
                logits.append(last_token_logits(model, outputs, mask, label_ids).detach().cpu())
        finally:
            for handle in handles:
                handle.remove()

    for key in states:
        states[key] = torch.cat(states[key], dim=0)

    if return_logits:
        return states, torch.cat(logits, dim=0)
    return states


@torch.no_grad()
def run_intervention(model, bank, sites, site_weights, source_states, bases, strength, label_ids, batch_size=32, return_logits=False):
    """Patch selected native dimensions or PCA directions and return answer logits/probabilities."""
    device = next(model.parameters()).device
    weights = torch.as_tensor(site_weights, dtype=torch.float32, device=device).flatten()
    if weights.numel() != len(sites):
        raise ValueError("site_weights must have one value per site")
    if not torch.isfinite(weights).all() or float(weights.sum().abs().item()) == 0.0:
        raise ValueError("site_weights must be finite with nonzero sum")
    weights = weights / weights.sum()
    print(strength * weights, len(weights))

    input_ids = bank["base_input_ids"]
    attention_mask = bank["base_attention_mask"]
    position_by_id = bank["base_position_by_id"]
    layer_ids, token_ids, outputs_all = [], [], []

    for L, token_id, _, _ in sites:
        L = int(L)
        if L not in layer_ids:
            layer_ids.append(L)
        if token_id not in token_ids:
            token_ids.append(token_id)
    layer_ids = sorted(layer_ids)

    for start in range(0, input_ids.shape[0], batch_size):
        end = min(start + batch_size, input_ids.shape[0])
        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        rows = torch.arange(ids.shape[0], device=device)
        pad_offset = (mask == 0).sum(dim=1)
        pos_by_token = {}
        for token_id in token_ids:
            pos_by_token[token_id] = pad_offset + position_by_id[token_id][start:end].to(device)

        handles = []

        def make_hook(layer_id):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_new = hidden
                changed = False

                for site_id, (L, token_id, a, b) in enumerate(sites):
                    L, a, b = int(L), int(a), int(b)
                    if L != layer_id:
                        continue
                    if not changed:
                        hidden_new = hidden.clone()
                        changed = True

                    key = (L, token_id)
                    pos = pos_by_token[token_id]
                    base_act = hidden_new[rows, pos, :].float()
                    source_act = source_states[key][start:end].to(device=device, dtype=torch.float32)
                    scale = float(strength) * weights[site_id]
                    # scale = float(strength)

                    if bases[key]["mode"] == "neuron":
                        patched = base_act.clone()
                        patched[:, a:b] = base_act[:, a:b] + scale * (source_act[:, a:b] - base_act[:, a:b])
                    else:
                        comps = bases[key]["components"].to(device=device, dtype=torch.float32)[a:b]
                        diff = source_act - base_act
                        patched = base_act + scale * ((diff @ comps.T) @ comps)

                    hidden_new[rows, pos, :] = patched.to(hidden_new.dtype)

                if not changed:
                    return None
                if isinstance(output, tuple):
                    return (hidden_new,) + output[1:]
                return hidden_new
            return hook

        for L in layer_ids:
            handles.append(model.model.layers[L].register_forward_hook(make_hook(L)))

        try:
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
            batch_logits = last_token_logits(model, outputs, mask, label_ids)
            if return_logits:
                outputs_all.append(batch_logits.detach().cpu())
            else:
                outputs_all.append(torch.softmax(batch_logits, dim=-1).detach().cpu())
        finally:
            for handle in handles:
                handle.remove()

    return torch.cat(outputs_all, dim=0)


def make_sites(layer_id, token_id, total_dim, resolution):
    """Split native-neuron or PCA coordinates into contiguous candidate sites."""
    sites = []
    for start in range(0, int(total_dim), int(resolution)):
        end = min(start + int(resolution), int(total_dim))
        sites.append((int(layer_id), token_id, int(start), int(end)))
    return sites


def site_signature(model, bank, sites, label_ids, mode="neuron", k=None, batch_size=32, strength=1.0, max_fit_states=4096, signature_method="family_mean", family_order=FAMILY_ORDER):
    """Build neural-site signatures from intervention logit deltas on D_ft."""
    base_states, base_logits = collect_site_activations(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_position_by_id"], sites, batch_size=batch_size, return_logits=True, label_ids=label_ids)
    source_states = collect_site_activations(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], sites, batch_size=batch_size)
    bases = fit_bases(sites, base_states, source_states, mode=mode, k=k, max_fit_states=max_fit_states)
    signatures = []

    for site in sites:
        patched_logits = run_intervention(model, bank, [site], [1.0], source_states, bases, strength, label_ids, batch_size=batch_size, return_logits=True)
        diff = patched_logits.float() - base_logits.float()
        signatures.append(make_signature(diff, method=signature_method, pair_source_families=bank["pair_source_families"], family_order=family_order))

    return {"sites": sites, "intervention_diff": torch.stack(signatures, dim=0), "bases": bases}


def top_sites_from_T(T, sites, var_id, top_k, min_mass=1e-8):
    """Select the top-k finite positive-mass sites from one OT coupling row."""
    valid_indices = []
    for i in range(len(sites)):
        value = T[var_id, i]
        if bool(torch.isfinite(value).item()) and float(value.detach().cpu().item()) > min_mass:
            valid_indices.append(i)
    if not valid_indices:
        raise ValueError(f"no positive-mass sites for var_id={var_id}")

    valid_scores = []
    for i in valid_indices:
        valid_scores.append(T[var_id, i])
    valid_scores = torch.stack(valid_scores)
    k = min(int(top_k), len(valid_indices))
    _, local_indices = torch.topk(valid_scores, k=k)

    selected_sites, selected_indices = [], []
    for local_i in local_indices.tolist():
        global_i = valid_indices[int(local_i)]
        selected_sites.append(sites[global_i])
        selected_indices.append(global_i)
    return selected_sites, selected_indices


def compute_iia(outputs, labels, var_name, pointer_num_labels=4):
    """Compute variable-specific interchange intervention accuracy."""
    scores = torch.as_tensor(outputs)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if var_name == "answer_pointer":
        scores = scores[:, :pointer_num_labels]
    elif var_name != "answer_token":
        raise ValueError(f"unknown var_name={var_name!r}")
    pred = scores.argmax(dim=-1).cpu()
    return (pred == labels.cpu()).float().mean().item()


def eval_intervention(model, bank, var_name, sites, weights, source_states, bases, strength, label_ids, batch_size=32):
    """Run one native/PCA intervention and return IIA for one target variable."""
    outputs = run_intervention(model, bank, sites, weights, source_states, bases, strength, label_ids, batch_size=batch_size)
    return compute_iia(outputs, bank["counterfactual_label_ids"][var_name], var_name)


def choose_stage_A_layers(model, names, cal_banks, coarse_sites, coarse_bases, T_coarse, label_ids, strength_values, top_k, keep_layers, threshold, batch_size=32):
    """Calibrate top coarse layers on D_cal and return validated Stage-A layers per variable."""
    top_layers, top_info, all_results = {}, {}, []

    for var_id, var_name in enumerate(names):
        cal_bank = cal_banks[var_name]
        top_sites, top_indices = top_sites_from_T(T_coarse, coarse_sites, var_id, top_k, min_mass=0.0)
        candidates = []
        print(f"\n[Stage A variable] {var_id} {var_name}")

        for rank, site in enumerate(top_sites, start=1):
            site_index = top_indices[rank - 1]
            L = int(site[0])
            mass = float(T_coarse[var_id, site_index].detach().cpu())
            source_states = collect_site_activations(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], [site], batch_size=batch_size)
            best = None

            for strength in strength_values:
                iia = eval_intervention(model, cal_bank, var_name, [site], [1.0], source_states, coarse_bases, float(strength), label_ids, batch_size=batch_size)
                result = {"var_id": var_id, "var_name": var_name, "raw_rank": rank, "site_index": site_index, "site": site, "layer": L, "coupling_mass": mass, "strength": float(strength), "cal_iia": float(iia)}
                all_results.append(result)
                if best is None or result["cal_iia"] > best["cal_iia"]:
                    best = result
                print(f"[Stage A CAL] var={var_name} layer={L} strength={strength} iia={iia:.4f}")
            candidates.append(best)

        passing = []
        for candidate in candidates:
            if candidate["cal_iia"] >= float(threshold):
                passing.append(candidate)
        if not passing:
            passing = candidates
            print(f"[Stage A fallback] var={var_name}: no layer passed threshold={threshold}")

        passing = sorted(passing, key=lambda x: x["cal_iia"], reverse=True)
        selected = passing[:int(keep_layers)]
        top_layers[var_id] = []
        for candidate in selected:
            top_layers[var_id].append(candidate["layer"])
        top_info[var_id] = selected
        print(f"[Stage A retained] var={var_name} layers={top_layers[var_id]}")

    return top_layers, top_info, all_results


def run_plot_progressive(model, tokenizer, layers, ft_size=128, cal_size=128, te_size=256, dataset_size=None, dataset_split="train", signature_method="family_mean", stage_A_mode="neuron", stage_B_mode="neuron", stage_A_k=None, stage_B_k=None, stage_A_eps=0.001, stage_B_eps=0.001, stage_A_top_layers=6, stage_A_keep_layers=1, stage_A_iia_threshold=0.7, resolutions=(128, 144, 192, 256, 288, 384, 576, 768), top_k_values=(1, 2, 3, 4, 5), strength_values=(1, 2, 4, 8, 16, 32, 64), stage_A_strength_values=None, stage_A_method="uot", stage_B_method="ot", chosen_token_position_id="last_token", device="cuda", seed=0, batch_size=32, max_fit_states=4096):
    """Run repo-style PLOT with independent neuron/PCA modes for Stage A and Stage B."""
    set_seed(seed)
    if stage_A_strength_values is None:
        stage_A_strength_values = strength_values

    label_ids = answer_label_ids(tokenizer)
    hidden_size = model.config.hidden_size
    layers = list(layers)
    stage_A_solver = get_solver(stage_A_method)
    stage_B_solver = get_solver(stage_B_method)

    ft_bank, cal_banks, te_banks = build_mcqa_banks(model=model, tokenizer=tokenizer, train_pool_size=ft_size, cal_size=cal_size, te_size=te_size, dataset_size=dataset_size, split=dataset_split, device=device, batch_size=batch_size, seed=seed)
    G_sig, names = variable_signature(ft_bank, num_labels=len(ANSWER_LETTERS), signature_method=signature_method, family_order=FAMILY_ORDER)
    print("[G]", G_sig.shape, names)

    n_ft = ft_bank["base_input_ids"].shape[0]
    coarse_dim = basis_dim(stage_A_mode, n_ft, hidden_size, stage_A_k, max_fit_states)
    coarse_sites = []
    for L in layers:
        coarse_sites.append((int(L), chosen_token_position_id, 0, coarse_dim))

    coarse_sig = site_signature(model, ft_bank, coarse_sites, label_ids, mode=stage_A_mode, k=stage_A_k, batch_size=batch_size, max_fit_states=max_fit_states, signature_method=signature_method, family_order=FAMILY_ORDER)
    S_coarse = coarse_sig["intervention_diff"]
    T_coarse = stage_A_solver(G_sig, S_coarse, eps=stage_A_eps)
    coarse_bases = coarse_sig["bases"]
    print("[Stage A]", "mode=", stage_A_mode, "dim=", coarse_dim, "shapes=", G_sig.shape, S_coarse.shape, T_coarse.shape)

    top_layers, top_coarse, stage_A_results = choose_stage_A_layers(model, names, cal_banks, coarse_sites, coarse_bases, T_coarse, label_ids, stage_A_strength_values, stage_A_top_layers, stage_A_keep_layers, stage_A_iia_threshold, batch_size=batch_size)

    top_layers = {
        0: [18], 1: [24]
    }

    fine_dim = basis_dim(stage_B_mode, n_ft, hidden_size, stage_B_k, max_fit_states)
    best_by_var, stage_B_results, fine_cache, cal_source_cache = {}, [], {}, {}

    for var_id, var_name in enumerate(names):
        cal_bank = cal_banks[var_name]
        stage_A_layers = top_layers[var_id]
        best = None
        print(f"\n[Stage B variable] {var_id} {var_name} layers={stage_A_layers} mode={stage_B_mode} dim={fine_dim}")

        for resolution in resolutions:
            layer_key = tuple(stage_A_layers)
            cache_key = (layer_key, int(resolution))

            if cache_key not in fine_cache:
                sites = []
                for L in stage_A_layers:
                    for site in make_sites(L, chosen_token_position_id, fine_dim, resolution):
                        sites.append(site)

                sig = site_signature(model, ft_bank, sites, label_ids, mode=stage_B_mode, k=stage_B_k, batch_size=batch_size, max_fit_states=max_fit_states, signature_method=signature_method, family_order=FAMILY_ORDER)
                S_fine = sig["intervention_diff"]
                T_fine = stage_B_solver(G_sig, S_fine, eps=stage_B_eps)
                fine_cache[cache_key] = {"sites": sites, "S": S_fine, "T": T_fine, "bases": sig["bases"]}

            cached = fine_cache[cache_key]
            sites, T_fine, bases = cached["sites"], cached["T"], cached["bases"]
            cal_cache_key = (var_name, cache_key)
            if cal_cache_key not in cal_source_cache:
                cal_source_cache[cal_cache_key] = collect_site_activations(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], sites, batch_size=batch_size)
            source_states = cal_source_cache[cal_cache_key]

            for top_k in top_k_values:
                selected_sites, selected_indices = top_sites_from_T(T_fine, sites, var_id, top_k)
                weights = T_fine[var_id, selected_indices].detach().float().cpu()
                for strength in strength_values:
                    iia = eval_intervention(model, cal_bank, var_name, selected_sites, weights, source_states, bases, float(strength), label_ids, batch_size=batch_size)
                    result = {"var_id": var_id, "var_name": var_name, "stage_A_layers": layer_key, "resolution": int(resolution), "top_k": len(selected_sites), "strength": float(strength), "cal_iia": float(iia), "selected_indices": selected_indices, "selected_sites": selected_sites, "selected_weights": weights, "cache_key": cache_key}
                    stage_B_results.append(result)
                    if best is None or result["cal_iia"] > best["cal_iia"]:
                        best = result
                    print(f"[Stage B CAL] var={var_name} layers={layer_key} resolution={resolution} top_k={len(selected_sites)} strength={strength} iia={iia:.4f}")

        best_by_var[var_id] = best
        print("\n[Stage B BEST]")
        print(best)

    test_results = {}
    for var_id, best in best_by_var.items():
        var_name = best["var_name"]
        te_bank = te_banks[var_name]
        cached = fine_cache[best["cache_key"]]
        source_states = collect_site_activations(model, te_bank["source_input_ids"], te_bank["source_attention_mask"], te_bank["source_position_by_id"], best["selected_sites"], batch_size=batch_size)
        test_iia = eval_intervention(model, te_bank, var_name, best["selected_sites"], best["selected_weights"], source_states, cached["bases"], best["strength"], label_ids, batch_size=batch_size)
        result = dict(best)
        result["test_iia"] = float(test_iia)
        test_results[var_id] = result
        print(f"\n[TEST] var={var_name} layers={best['stage_A_layers']} resolution={best['resolution']} top_k={best['top_k']} strength={best['strength']} test_iia={test_iia:.4f}")

    return {"names": names, "G": G_sig, "S_coarse": S_coarse, "T_coarse": T_coarse, "top_layers_var": top_layers, "top_coarse_by_var": top_coarse, "stage_A_cal_results": stage_A_results, "best_by_var": best_by_var, "stage_B_cal_results": stage_B_results, "test_results": test_results, "fine_cache": fine_cache, "signature_method": signature_method, "stage_A_mode": stage_A_mode, "stage_B_mode": stage_B_mode}


if __name__ == "__main__":
    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device
    layers = []
    for L in range(model.config.num_hidden_layers):
        layers.append(L)

    results = run_plot_progressive(
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        ft_size=200,
        cal_size=100,
        te_size=100,
        dataset_size=None,
        dataset_split="train",
        signature_method="family_mean",  # "family_mean" or "concat"
        stage_A_mode="neuron",
        stage_B_mode="neuron",
        stage_A_k=None,
        stage_B_k=None,
        stage_A_eps=1,
        stage_B_eps=1,
        stage_A_method="uot",
        stage_B_method="ot",
        stage_A_top_layers=6,
        stage_A_keep_layers=1,
        stage_A_iia_threshold=0.7,
        resolutions=(128, 144, 192, 256, 288, 384, 576, 768),
        top_k_values=(1, 2, 4),
        strength_values=(0.5, 1, 2, 4),
        chosen_token_position_id="last_token",
        device=device,
        seed=2,
        batch_size=32,
    )

    os.makedirs("results", exist_ok=True)
    save_path = "results/plot_progressive_last_token.pt"
    torch.save(results, save_path)
    print("saved to:", save_path)
    print("\n===== FINAL TEST RESULTS =====")

    for var_id in results["test_results"]:
        r = results["test_results"][var_id]
        print("var_id=", var_id, "var_name=", r["var_name"], "stage_A_layers=", r["stage_A_layers"], "resolution=", r["resolution"], "top_k=", r["top_k"], "strength=", r["strength"], "cal_iia=", r["cal_iia"], "test_iia=", r["test_iia"])
