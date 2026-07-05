import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist


from mcqa_data import build_bank, letter_token_id
from mcqa_neural_net import load_gemma_model

from mcqa_ot import solve_ot, solve_uot

from sklearn.decomposition import PCA

import string


def variable_signature(base_output, dict_cf_outputs, num_labels=None):

    base_onehot = F.one_hot(torch.as_tensor(base_output, dtype=torch.long), num_labels).float()

    var_signatures, names = [], []
    for var_name, cf_output in dict_cf_outputs.items():

        cf_onehot = F.one_hot(torch.as_tensor(cf_output, dtype=torch.long), num_labels).float()
        var_signatures.append((cf_onehot - base_onehot).reshape(-1))
        names.append(var_name)

    return torch.stack(var_signatures, dim=0), names



def max_pca_dim_for_ft(ft_size, hidden_size, k=None):
    """
    PCA fit trên base + source nên số components tối đa <= 2 * ft_size.
    """
    max_fit_states = 2 * int(ft_size)

    if k is None:
        return min(max_fit_states, hidden_size)

    return min(int(k), max_fit_states, hidden_size)


def make_pca_prefix_bands(ft_size, hidden_size, k=None):
    """
    Prefix PCA bands:
        (0,1), (0,2), (0,4), ..., (0,max_dim)

    Dùng để hỏi: top d PCA components đầu có đủ causal direction không?
    """
    max_dim = max_pca_dim_for_ft(
        ft_size=ft_size,
        hidden_size=hidden_size,
        k=k,
    )

    dims = []
    d = 1
    while d < max_dim:
        dims.append(d)
        d *= 2

    dims.append(max_dim)

    return [(0, int(d)) for d in dims]



def make_pca_equal_bands_for_count(pca_rank, band_count):
    import numpy as np

    edges = np.linspace(0, pca_rank, band_count + 1)
    edges = np.round(edges).astype(int).tolist()

    bands = []

    for i in range(band_count):
        a = edges[i]
        b = edges[i + 1]

        if a < b:
            bands.append((int(a), int(b)))

    return bands


def make_pca_equal_bands(ft_size, hidden_size, k=None, num_bands=8):
    """
    Disjoint equal PCA bands:
        (0,32), (32,64), ...

    Dùng nếu muốn sites không overlap.
    """
    max_dim = max_pca_dim_for_ft(
        ft_size=ft_size,
        hidden_size=hidden_size,
        k=k,
    )

    edges = np.linspace(0, max_dim, num_bands + 1)
    edges = np.round(edges).astype(int).tolist()

    bands = []
    for i in range(num_bands):
        a, b = edges[i], edges[i + 1]
        if a < b:
            bands.append((int(a), int(b)))

    return bands


def make_neuron_bands(hidden_size, band_size=256):
    return [
        (start, min(start + band_size, hidden_size))
        for start in range(0, hidden_size, band_size)
    ]



def last_real_token_positions(attention_mask):
    idx = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return (attention_mask * idx.unsqueeze(0)).max(dim=1).values



def precompute_basis_for_each_layer(
    model,
    sites,
    base_states,
    source_states,
    mode="pca",
    k=512,
    max_fit_states=4096,
):
    """
    Returns:
        bases[(layer_id, token_id)] = basis
    """
    keys = sorted(set((int(L), token_id) for L, token_id, _, _ in sites))
    bases = {}

    for key in keys:
        X = torch.cat([base_states[key], source_states[key]], dim=0).float()
        n_samples, hidden_size = X.shape

        if mode == "pca":

            if max_fit_states is not None and len(X) > max_fit_states:
                idx = torch.randperm(len(X))[:max_fit_states]
                X = X[idx]

            k_eff = min(k, n_samples, hidden_size)

            pca = PCA(n_components=k_eff, whiten=False)
            pca.fit(X.numpy())

            bases[key] = {
                "mode": "pca",
                "mean": torch.tensor(pca.mean_, dtype=torch.float32),  # [H]
                "components": torch.tensor(pca.components_, dtype=torch.float32),  # [K, H]
            }

        elif mode == "neuron":
            bases[key] = {
                "mode": "neuron",
                "mean": torch.zeros(hidden_size, dtype=torch.float32),  # [H]
                "components": torch.eye(hidden_size, dtype=torch.float32),  # [H, H]
            }

    return bases



def answer_probs_from_model_output(
    model,
    outputs,
    attention_mask,
    answer_label_ids=None,
):
    device = next(model.parameters()).device

    B = attention_mask.shape[0]
    rows = torch.arange(B, device=device)

    # last_real_token_positions
    idx = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    logit_pos = (attention_mask * idx.unsqueeze(0)).max(dim=1).values

    hidden_at_pos = outputs.last_hidden_state[rows, logit_pos, :]

    if answer_label_ids is None:
        logits = model.lm_head(hidden_at_pos)

    else:
        answer_ids = torch.tensor(
            answer_label_ids,
            device=device,
            dtype=torch.long,
        )

        W = model.lm_head.weight[answer_ids].to(hidden_at_pos.dtype)
        logits = hidden_at_pos @ W.T

        bias = getattr(model.lm_head, "bias", None)

        if bias is not None:
            logits = logits + bias[answer_ids]

        softcap = getattr(model.config, "final_logit_softcapping", None)

        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap

    probs = torch.softmax(logits.float(), dim=-1)

    return probs


@torch.no_grad()
def collect_site_activations(
    model,
    input_ids,
    attention_mask,
    position_by_id,
    sites,
    batch_size=32,
    return_output=False,
    answer_label_ids=None,
    output_dtype=torch.float16,
):
    device = next(model.parameters()).device
    N = input_ids.shape[0]

    layer_ids = []
    token_ids = []
    collect_keys = []
    collected = {}
    collected_outputs = []

    for site in sites:
        L, token_id, start_dim, end_dim = site
        L = int(L)

        if L not in layer_ids:
            layer_ids.append(L)

        if token_id not in token_ids:
            token_ids.append(token_id)

        key = (L, token_id)

        if key not in collected:
            collected[key] = []
            collect_keys.append(key)

    layer_ids = sorted(layer_ids)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)

        B = ids.shape[0]
        rows = torch.arange(B, device=device)

        pad_offset = (mask == 0).sum(dim=1)

        pos_by_token = {}

        for token_id in token_ids:
            raw_pos = position_by_id[token_id][start:end].to(device)
            pos_by_token[token_id] = pad_offset + raw_pos

        handles = []

        def make_hook(layer_id):
            def hook(_module, _inputs, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output

                for key in collect_keys:
                    L_collect, token_id_collect = key

                    if L_collect != layer_id:
                        continue

                    pos = pos_by_token[token_id_collect]
                    act = hidden[rows, pos, :]

                    collected[key].append(
                        act.detach().float().cpu()
                    )

            return hook

        for layer_id in layer_ids:
            handle = model.model.layers[layer_id].register_forward_hook(
                make_hook(layer_id)
            )
            handles.append(handle)

        try:
            outputs = model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0),
                use_cache=False,
                return_dict=True,
            )

            if return_output:
                probs = answer_probs_from_model_output(
                    model=model,
                    outputs=outputs,
                    attention_mask=mask,
                    answer_label_ids=answer_label_ids,
                )

                collected_outputs.append(
                    probs.detach().to(dtype=output_dtype, device="cpu")
                )

        finally:
            for handle in handles:
                handle.remove()

    states = {}

    for key in collected:
        states[key] = torch.cat(collected[key], dim=0)

    if return_output:
        outputs = torch.cat(collected_outputs, dim=0)
        return states, outputs

    return states



@torch.no_grad()
def run_intervention(
    model,
    input_ids,
    attention_mask,
    position_by_id,
    sites,
    site_weights,
    source_states,
    bases,
    strength=1.0,
    batch_size=32,
    answer_label_ids=None,
    output_dtype=torch.float16,
):
    device = next(model.parameters()).device
    N = input_ids.shape[0]

    layer_ids = []
    token_ids = []
    collected_outputs = []

    for site in sites:
        L, token_id, start_dim, end_dim = site
        L = int(L)

        if L not in layer_ids:
            layer_ids.append(L)

        if token_id not in token_ids:
            token_ids.append(token_id)

    layer_ids = sorted(layer_ids)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)

        B = ids.shape[0]
        rows = torch.arange(B, device=device)

        pad_offset = (mask == 0).sum(dim=1)

        pos_by_token = {}

        for token_id in token_ids:
            raw_pos = position_by_id[token_id][start:end].to(device)
            pos_by_token[token_id] = pad_offset + raw_pos

        handles = []

        def make_hook(layer_id):
            def hook(_module, _inputs, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output

                hidden_new = hidden
                changed = False

                for site_idx in range(len(sites)):
                    site = sites[site_idx]

                    L, token_id, start_dim, end_dim = site
                    L = int(L)

                    if L != layer_id:
                        continue

                    if not changed:
                        hidden_new = hidden.clone()
                        changed = True

                    key = (L, token_id)
                    pos = pos_by_token[token_id]

                    # Current activation on the current trajectory
                    base_acts = hidden_new[rows, pos, :].float()

                    source_acts = source_states[key][start:end]
                    source_acts = source_acts.to(
                        device=device,
                        dtype=torch.float32,
                    )

                    basis = bases[key]

                    # Coupling-weighted intervention strength
                    site_weight = float(site_weights[site_idx])

                    effective_strength = (
                        float(strength) * site_weight
                    )

                    if basis["mode"] == "pca":
                        comps = basis["components"].to(
                            device=device,
                            dtype=torch.float32,
                        )

                        selected_comps = comps[start_dim:end_dim]

                        diff_x = source_acts - base_acts
                        diff_z = diff_x @ selected_comps.T
                        delta_x = diff_z @ selected_comps

                        patched_acts = (
                            base_acts
                            + effective_strength * delta_x
                        )

                    elif basis["mode"] == "neuron":
                        patched_acts = base_acts.clone()

                        delta = (
                            source_acts[:, start_dim:end_dim]
                            - base_acts[:, start_dim:end_dim]
                        )

                        patched_acts[:, start_dim:end_dim] = (
                            base_acts[:, start_dim:end_dim]
                            + effective_strength * delta
                        )

                    hidden_new[rows, pos, :] = patched_acts.to(
                        hidden_new.dtype
                    )

                if changed:
                    if isinstance(output, tuple):
                        return (hidden_new,) + output[1:]

                    return hidden_new

                return None

            return hook

        for layer_id in layer_ids:
            handle = model.model.layers[
                layer_id
            ].register_forward_hook(
                make_hook(layer_id)
            )

            handles.append(handle)

        try:
            outputs = model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=(
                    mask.long().cumsum(dim=-1) - 1
                ).clamp(min=0),
                use_cache=False,
                return_dict=True,
            )

            probs = answer_probs_from_model_output(
                model=model,
                outputs=outputs,
                attention_mask=mask,
                answer_label_ids=answer_label_ids,
            )

            collected_outputs.append(
                probs.detach().to(
                    dtype=output_dtype,
                    device="cpu",
                )
            )

        finally:
            for handle in handles:
                handle.remove()

    outputs = torch.cat(collected_outputs, dim=0)

    return outputs



def make_native_sites_for_layer(
    layer_id,
    token_id,
    hidden_size,
    resolution,
):
    sites = []

    for start_dim in range(0, hidden_size, int(resolution)):
        end_dim = min(start_dim + int(resolution), hidden_size)

        site = (
            int(layer_id),
            token_id,
            int(start_dim),
            int(end_dim),
        )

        sites.append(site)

    return sites



@torch.no_grad()
def site_signature(
    model,
    bank,
    sites,
    answer_label_ids=None,
    mode="pca",
    k=512,
    strength=1.0,
    batch_size=16,
    max_fit_states=4096,
    output_dtype=torch.float16,
    return_bases=False,
):
    base_input_ids = bank["base_input_ids"]
    base_attention_mask = bank["base_attention_mask"]

    source_input_ids = bank["source_input_ids"]
    source_attention_mask = bank["source_attention_mask"]

    num_sites = len(sites)
    N = base_input_ids.shape[0]

    base_states, base_output = collect_site_activations(
        model=model,
        input_ids=base_input_ids,
        attention_mask=base_attention_mask,
        position_by_id=bank["base_position_by_id"],
        sites=sites,
        batch_size=batch_size,
        return_output=True,
        answer_label_ids=answer_label_ids,
        output_dtype=output_dtype,
    )

    source_states = collect_site_activations(
        model=model,
        input_ids=source_input_ids,
        attention_mask=source_attention_mask,
        position_by_id=bank["source_position_by_id"],
        sites=sites,
        batch_size=batch_size,
        return_output=False,
        answer_label_ids=answer_label_ids,
        output_dtype=output_dtype,
    )

    bases = precompute_basis_for_each_layer(
        model=model,
        sites=sites,
        base_states=base_states,
        source_states=source_states,
        mode=mode,
        k=k,
        max_fit_states=max_fit_states,
    )

    num_answers = base_output.shape[-1]
    output_width = N * num_answers

    intervened_output = torch.empty(
        (num_sites, output_width),
        dtype=output_dtype,
        device="cpu",
    )

    intervention_diff = torch.empty(
        (num_sites, output_width),
        dtype=output_dtype,
        device="cpu",
    )

    for site_idx in range(num_sites):
        site = sites[site_idx]

        patched_output = run_intervention(
            model=model,
            input_ids=base_input_ids,
            attention_mask=base_attention_mask,
            position_by_id=bank["base_position_by_id"],
            sites=[site],
            site_weights=[1.0],
            source_states=source_states,
            bases=bases,
            strength=strength,
            batch_size=batch_size,
            answer_label_ids=answer_label_ids,
            output_dtype=output_dtype,
        )

        diff = patched_output.float() - base_output.float()

        intervened_output[site_idx] = patched_output.reshape(-1).to(output_dtype)
        intervention_diff[site_idx] = diff.reshape(-1).to(output_dtype)

    result = {
        "sites": sites,
        "base_output": base_output,
        "intervened_output": intervened_output,
        "intervention_diff": intervention_diff,
    }

    if return_bases:
        result["bases"] = bases

    return result



def select_top_sites_from_T(T, sites, var_id, top_k):
    scores = T[var_id]
    top_k = min(int(top_k), len(sites))

    values, indices = torch.topk(scores, k=top_k)

    selected_sites = []
    selected_indices = []

    for idx in indices.tolist():
        selected_sites.append(sites[int(idx)])
        selected_indices.append(int(idx))

    return selected_sites, selected_indices




def compute_iia(output_probs, target_labels):
    pred = output_probs.argmax(dim=-1).cpu()
    target = torch.as_tensor(target_labels, dtype=torch.long).cpu()

    return (pred == target).float().mean().item()


def evaluate_stage_B_candidate(
    model,
    var_id,
    var_name,
    stage_A_layers,
    resolution,
    top_k,
    strength,
    G_sig,
    ft_bank,
    cal_bank,
    fine_cache,
    stage_B_solver,
    stage_B_eps,
    answer_label_ids,
    chosen_token_position_id,
    hidden_size,
    output_dtype=torch.float16,
    batch_size=32,
):
    """
    Evaluate one Stage-B configuration:
        (resolution, top_k, strength)

    Returns:
        candidate: dict
    """

    # ========================================================
    # 1. Build cache key
    # ========================================================
    layer_key_list = []

    for L in stage_A_layers:
        layer_key_list.append(int(L))

    layer_key = tuple(layer_key_list)

    cache_key = (
        layer_key,
        int(resolution),
    )

    # ========================================================
    # 2. Build fine candidate sites and OT once per
    #    (stage_A_layers, resolution)
    # ========================================================
    if cache_key not in fine_cache:
        sites_fine = []

        for L in stage_A_layers:
            layer_sites = make_native_sites_for_layer(
                layer_id=int(L),
                token_id=chosen_token_position_id,
                hidden_size=hidden_size,
                resolution=resolution,
            )

            for site in layer_sites:
                sites_fine.append(site)

        sig_fine = site_signature(
            model=model,
            bank=ft_bank,
            sites=sites_fine,
            answer_label_ids=answer_label_ids,
            mode="neuron",
            k=None,
            strength=1.0,
            batch_size=batch_size,
            output_dtype=output_dtype,
            return_bases=True,
        )

        S_fine_sig = sig_fine["intervention_diff"]

        T_fine = stage_B_solver(
            G_sig,
            S_fine_sig,
            eps=stage_B_eps,
        )

        fine_cache[cache_key] = {
            "stage_A_layers": layer_key,
            "sites_fine": sites_fine,
            "T_fine": T_fine,
            "S_fine": S_fine_sig,
            "bases": sig_fine["bases"],
        }

    # ========================================================
    # 3. Read cached Stage-B objects
    # ========================================================
    cached = fine_cache[cache_key]

    sites_fine = cached["sites_fine"]
    T_fine = cached["T_fine"]
    bases_fine = cached["bases"]

    # ========================================================
    # 4. Select top-k sites
    # ========================================================
    selected_sites, selected_indices = select_top_sites_from_T(
        T=T_fine,
        sites=sites_fine,
        var_id=var_id,
        top_k=top_k,
    )

    # ========================================================
    # 5. Collect source activations on D_cal
    # ========================================================
    cal_source_states = collect_site_activations(
        model=model,
        input_ids=cal_bank["source_input_ids"],
        attention_mask=cal_bank["source_attention_mask"],
        position_by_id=cal_bank["source_position_by_id"],
        sites=selected_sites,
        batch_size=batch_size,
        return_output=False,
        answer_label_ids=answer_label_ids,
        output_dtype=output_dtype,
    )

    # ========================================================
    # 6. Run intervention
    # ========================================================
    selected_weights = T_fine[var_id, selected_indices]

    selected_weights = selected_weights.detach().float().cpu()

    cal_output = run_intervention(
        model=model,
        input_ids=cal_bank["base_input_ids"],
        attention_mask=cal_bank["base_attention_mask"],
        position_by_id=cal_bank["base_position_by_id"],
        sites=selected_sites,
        site_weights=selected_weights,
        source_states=cal_source_states,
        bases=bases_fine,
        strength=float(strength),
        batch_size=32,
        answer_label_ids=answer_label_ids,
        output_dtype=torch.float16,
    )

    # ========================================================
    # 7. Compute calibration IIA
    # ========================================================
    cal_iia = compute_iia(
        output_probs=cal_output,
        target_labels=cal_bank[
            "counterfactual_label_ids"
        ][var_name],
    )

    # ========================================================
    # 8. Return candidate metadata
    # ========================================================
    candidate = {
        "var_id": int(var_id),
        "var_name": var_name,
        "stage_A_layers": layer_key,
        "resolution": int(resolution),
        "top_k": int(top_k),
        "strength": float(strength),
        "cal_iia": float(cal_iia),
        "selected_indices": selected_indices,
        "selected_sites": selected_sites,
        "cache_key": cache_key,
    }

    return candidate



def run_plot_progressive(
    model,
    tokenizer,
    target_variable,
    layers,
    ft_size=128,
    cal_size=128,
    te_size=256,
    stage_A_eps=0.001,
    stage_B_eps=0.001,
    stage_A_top_layers=6,
    resolutions=(128, 144, 192, 256, 288, 384, 576, 768),
    top_k_values=range(1, 6),
    strength_values=(1, 2, 4, 8, 16, 32, 64),
    stage_A_strength_values=None,
    stage_A_method="ot",
    stage_B_method="ot",
    chosen_token_position_id="correct_symbol",
    device="cuda",
):

    answer_letters = tuple(string.ascii_uppercase)

    answer_label_ids = []
    for L in answer_letters:
        answer_label_ids.append(letter_token_id(tokenizer, L))

    num_labels = len(answer_letters)
    hidden_size = model.config.hidden_size
    layers = list(layers)

    if stage_A_strength_values is None:
        stage_A_strength_values = strength_values

    if stage_A_method == "ot":
        stage_A_solver = solve_ot
    else:
        stage_A_solver = solve_uot

    if stage_B_method == "ot":
        stage_B_solver = solve_ot
    else:
        stage_B_solver = solve_uot

    # ============================================================
    # 1. Build banks
    # ============================================================
    ft_bank = build_bank(
        model=model,
        tokenizer=tokenizer,
        target_variable=target_variable,
        bank_size=ft_size,
        answer_letters=answer_letters,
        device=device,
        example_offset=0,
    )

    cal_bank = build_bank(
        model=model,
        tokenizer=tokenizer,
        target_variable=target_variable,
        bank_size=cal_size,
        answer_letters=answer_letters,
        device=device,
        example_offset=ft_size * 3,
    )

    te_bank = build_bank(
        model=model,
        tokenizer=tokenizer,
        target_variable=target_variable,
        bank_size=te_size,
        answer_letters=answer_letters,
        device=device,
        example_offset=ft_size * 3 + cal_size * 3,
    )

    # ============================================================
    # 2. Variable signatures on D_ft
    # ============================================================
    G_sig, names = variable_signature(
        base_output=ft_bank["base_answer_label_ids"],
        dict_cf_outputs=ft_bank["counterfactual_label_ids"],
        num_labels=num_labels,
    )

    print("[G]", G_sig.shape, names)

    # ============================================================
    # 3. Stage A: coarse full-layer OT on D_ft
    # ============================================================
    coarse_sites = []

    for L in layers:
        site = (int(L), chosen_token_position_id, 0, hidden_size)
        coarse_sites.append(site)

    sig_coarse = site_signature(
        model=model,
        bank=ft_bank,
        sites=coarse_sites,
        answer_label_ids=answer_label_ids,
        mode="neuron",
        k=None,
        strength=1.0,
        batch_size=32,
        output_dtype=torch.float16,
        return_bases=True,
    )
    
    S_coarse_sig = sig_coarse["intervention_diff"]

    T_coarse = stage_A_solver(
        G_sig,
        S_coarse_sig,
        eps=stage_A_eps,
    )

    coarse_bases = sig_coarse["bases"]

    print("[Stage A shapes]", G_sig.shape, S_coarse_sig.shape, T_coarse.shape)


    # ============================================================
    # 4. Stage A selection:
    #    keep top-K raw layers directly from T_coarse
    # ============================================================
    top_layers_var = {}
    top_coarse_by_var = {}

    for var_id in range(len(names)):
        var_name = names[var_id]

        top_sites, top_indices = select_top_sites_from_T(
            T=T_coarse,
            sites=coarse_sites,
            var_id=var_id,
            top_k=stage_A_top_layers,
        )

        selected_layers = []
        selected_candidates = []

        print()
        print("[Stage A variable]", var_id, var_name)

        for site_pos in range(len(top_sites)):
            site = top_sites[site_pos]
            site_index = top_indices[site_pos]

            L, token_id, start_dim, end_dim = site
            L = int(L)
            site_index = int(site_index)

            coupling_mass = float(
                T_coarse[var_id, site_index].detach().cpu()
            )

            candidate = {
                "var_id": int(var_id),
                "var_name": var_name,
                "site_index": site_index,
                "site": site,
                "layer": L,
                "coupling_mass": coupling_mass,
            }

            selected_layers.append(L)
            selected_candidates.append(candidate)

            print(
                "[Stage A SELECT]",
                "var=", var_name,
                "rank=", site_pos + 1,
                "layer=", L,
                "mass=", coupling_mass,
            )

        top_layers_var[var_id] = selected_layers
        top_coarse_by_var[var_id] = selected_candidates


    print()
    print("[Stage A raw top_layers_var]", top_layers_var)


    # ============================================================
    # 5. Stage B: fine multi-layer native search
    # ============================================================
    best_by_var = {}
    stage_B_cal_results = []
    fine_cache = {}

    for var_id in range(len(names)):
        var_name = names[var_id]
        stage_A_layers = top_layers_var[var_id]

        best_candidate = None

        print()
        print("[Stage B variable]", var_id, var_name)
        print("[Stage B Stage-A layers]", stage_A_layers)

        for resolution in resolutions:
            for top_k in top_k_values:
                for strength in strength_values:

                    candidate = evaluate_stage_B_candidate(
                        model=model,
                        var_id=var_id,
                        var_name=var_name,
                        stage_A_layers=stage_A_layers,
                        resolution=resolution,
                        top_k=top_k,
                        strength=strength,
                        G_sig=G_sig,
                        ft_bank=ft_bank,
                        cal_bank=cal_bank,
                        fine_cache=fine_cache,
                        stage_B_solver=stage_B_solver,
                        stage_B_eps=stage_B_eps,
                        answer_label_ids=answer_label_ids,
                        chosen_token_position_id=chosen_token_position_id,
                        hidden_size=hidden_size,
                        output_dtype=torch.float16,
                        batch_size=32,
                    )

                    stage_B_cal_results.append(candidate)

                    if best_candidate is None:
                        best_candidate = candidate
                    else:
                        if candidate["cal_iia"] > best_candidate["cal_iia"]:
                            best_candidate = candidate

                    print(
                        "[Stage B CAL]",
                        "var=", var_name,
                        "layers=", candidate["stage_A_layers"],
                        "resolution=", candidate["resolution"],
                        "top_k=", candidate["top_k"],
                        "strength=", candidate["strength"],
                        "iia=", round(candidate["cal_iia"], 4),
                    )

        best_by_var[var_id] = best_candidate

        print()
        print("[Stage B BEST]")
        print(best_candidate)


    # ============================================================
    # 6. Final test on D_te
    # ============================================================
    test_results = {}

    for var_id in range(len(names)):
        best_candidate = best_by_var[var_id]

        var_name = best_candidate["var_name"]
        stage_A_layers = best_candidate["stage_A_layers"]
        resolution = best_candidate["resolution"]
        selected_sites = best_candidate["selected_sites"]
        strength = best_candidate["strength"]

        cache_key = (
            tuple(stage_A_layers),
            int(resolution),
        )

        bases_fine = fine_cache[cache_key]["bases"]

        # --------------------------------------------------------
        # Collect source activations from all selected sites
        # potentially across multiple layers
        # --------------------------------------------------------
        te_source_states = collect_site_activations(
            model=model,
            input_ids=te_bank["source_input_ids"],
            attention_mask=te_bank["source_attention_mask"],
            position_by_id=te_bank["source_position_by_id"],
            sites=selected_sites,
            batch_size=32,
            return_output=False,
            answer_label_ids=answer_label_ids,
            output_dtype=torch.float16,
        )

        # --------------------------------------------------------
        # Sequential multi-layer intervention should happen
        # inside run_intervention()
        # --------------------------------------------------------
        te_output = run_intervention(
            model=model,
            input_ids=te_bank["base_input_ids"],
            attention_mask=te_bank["base_attention_mask"],
            position_by_id=te_bank["base_position_by_id"],
            sites=selected_sites,
            source_states=te_source_states,
            bases=bases_fine,
            strength=float(strength),
            batch_size=32,
            answer_label_ids=answer_label_ids,
            output_dtype=torch.float16,
        )

        test_iia = compute_iia(
            output_probs=te_output,
            target_labels=te_bank[
                "counterfactual_label_ids"
            ][var_name],
        )

        final_candidate = dict(best_candidate)
        final_candidate["test_iia"] = float(test_iia)

        test_results[var_id] = final_candidate

        print()
        print(
            "[TEST]",
            "var=", var_name,
            "stage_A_layers=", stage_A_layers,
            "resolution=", resolution,
            "top_k=", best_candidate["top_k"],
            "strength=", strength,
            "selected_sites=", selected_sites,
            "test_iia=", round(float(test_iia), 4),
        )


    return {
        "names": names,
        "G": G_sig,
        "T_coarse": T_coarse,
        "S_coarse": S_coarse_sig,
        "top_layers_var": top_layers_var,
        "top_coarse_by_var": top_coarse_by_var,
        "best_by_var": best_by_var,
        "test_results": test_results,
        "stage_B_cal_results": stage_B_cal_results,
        "fine_cache": fine_cache,
    }




if __name__ == "__main__":
    import os
    import torch

    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device

    print("device:", device)
    print("hidden_size:", model.config.hidden_size)
    print("num_layers:", model.config.num_hidden_layers)

    layers = []

    for L in range(model.config.num_hidden_layers):
        layers.append(L)

    results = run_plot_progressive(
        model=model,
        tokenizer=tokenizer,
        target_variable=("answer_pointer", "answer_token"),
        layers=layers,

        ft_size=128,
        cal_size=128,
        te_size=256,

        stage_A_eps=0.03,
        stage_B_eps=0.03,
        stage_A_method="uot",
        stage_B_method="ot",

        stage_A_top_layers=2,

        resolutions=(128, 144, 192, 256, 288, 384, 576, 768),
        top_k_values=(1, 2, 3, 4, 5),
        strength_values=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4, 8, 16, 32, 64),

        chosen_token_position_id="last_token",
        device=device,
    )

    os.makedirs("results", exist_ok=True)

    save_path = "results/plot_progressive_last_token.pt"

    torch.save(
        results,
        save_path,
    )

    print("saved to:", save_path)

    print()
    print("===== FINAL TEST RESULTS =====")

    for var_id in results["test_results"]:
        r = results["test_results"][var_id]

        print(
            "var_id=", var_id,
            "var_name=", r["var_name"],
            "layer=", r["best_layer"],
            "resolution=", r["resolution"],
            "top_k=", r["top_k"],
            "strength=", r["strength"],
            "cal_iia=", r["cal_iia"],
            "test_iia=", r["test_iia"],
        )
