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


def make_position_ids(attention_mask):
    return (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)


def to_padded_positions(attention_mask, positions, positions_are_padded=False):
    if positions_are_padded:
        return positions
    return (attention_mask == 0).sum(dim=1) + positions


def last_real_token_positions(attention_mask):
    idx = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return (attention_mask * idx.unsqueeze(0)).max(dim=1).values


def get_site_keys(sites):
    return sorted(set((int(L), token_id) for L, token_id, _, _ in sites))


def answer_logits_from_hidden(model, hidden_at_pos, answer_label_ids):
    """
    hidden_at_pos: [B, hidden_size]
    answer_label_ids: list[int], token ids của A/B/C/D...

    Returns:
        logits: [B, num_answer_letters]
    """
    device = hidden_at_pos.device

    answer_ids = torch.tensor(
        answer_label_ids,
        device=device,
        dtype=torch.long,
    )

    # lm_head.weight: [vocab_size, hidden_size]
    W = model.lm_head.weight[answer_ids].to(dtype=hidden_at_pos.dtype)
    logits = hidden_at_pos @ W.T

    bias = getattr(model.lm_head, "bias", None)
    if bias is not None:
        logits = logits + bias[answer_ids]

    # Gemma-2 có thể có final logit softcapping
    softcap = getattr(model.config, "final_logit_softcapping", None)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap

    return logits



@torch.no_grad()
def collect_site_activations(
    model,
    input_ids,
    attention_mask,
    position_by_id,
    sites,
    batch_size=32,
    positions_are_padded=False,
    return_output=False,
    output_dtype=torch.float16,
    answer_label_ids=None,
):
    device = next(model.parameters()).device
    token_id = sites[0][1]
    layer_ids = sorted(set(int(L) for L, _, _, _ in sites))
    collected = {
        (L, token_id): []
        for L in layer_ids
    }
    collected_outputs = []
    N = input_ids.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        B = ids.shape[0]
        rows = torch.arange(B, device=device)
        pos = position_by_id[token_id][start:end].to(device)
        pos = to_padded_positions(
            attention_mask=mask,
            positions=pos,
            positions_are_padded=positions_are_padded,
        )
        handles = []
        def make_hook(layer_id):
            key = (int(layer_id), token_id)
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                act = hidden[rows, pos, :]
                collected[key].append(act.detach().float().cpu())
            return hook
        for layer_id in layer_ids:
            h = model.model.layers[layer_id].register_forward_hook(
                make_hook(layer_id)
            )
            handles.append(h)
        try:
            outputs = model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=make_position_ids(mask),
                use_cache=False,
                return_dict=True,
            )
            if return_output:
                logit_pos = last_real_token_positions(mask)
                hidden_at_pos = outputs.last_hidden_state[rows, logit_pos, :]
                if answer_label_ids is None:
                    logits = model.lm_head(hidden_at_pos)
                else:
                    logits = answer_logits_from_hidden(
                        model=model,
                        hidden_at_pos=hidden_at_pos,
                        answer_label_ids=answer_label_ids,
                    )
                collected_outputs.append(
                    logits.detach().to(dtype=output_dtype, device="cpu")
                )
        finally:
            for h in handles:
                h.remove()
    states = {
        key: torch.cat(chunks, dim=0)
        for key, chunks in collected.items()
    }
    if return_output:
        outputs = torch.cat(collected_outputs, dim=0)
        return states, outputs
    return states



def fit_pca_basis(base_act, source_act, k=512, max_fit_states=4096):
    """
    base_act:   [N, H]
    source_act: [N, H]

    Returns:
        basis dict
    """
    X = torch.cat([base_act, source_act], dim=0).float()

    if max_fit_states is not None and len(X) > max_fit_states:
        idx = torch.randperm(len(X))[:max_fit_states]
        X = X[idx]

    n_samples, hidden_size = X.shape
    k_eff = min(k, n_samples, hidden_size)

    pca = PCA(n_components=k_eff, whiten=False)
    pca.fit(X.numpy())

    return {
        "mode": "pca",
        "components": torch.tensor(pca.components_, dtype=torch.float32),  # [K, H]
        "explained_variance": torch.tensor(pca.explained_variance_, dtype=torch.float32),
    }


def fit_basis_for_sites(
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
    keys = get_site_keys(sites)
    bases = {}

    for key in keys:
        if mode == "pca":
            bases[key] = fit_pca_basis(
                base_act=base_states[key],
                source_act=source_states[key],
                k=k,
                max_fit_states=max_fit_states,
            )

        elif mode == "neuron":
            bases[key] = {"mode": "neuron"}

        else:
            raise ValueError(f"Unknown mode: {mode}")

    return bases


@torch.no_grad()
def precompute_delta_states(
    base_states,
    source_states,
    bases,
    batch_size=4096,
    cache_dtype=torch.float16,
):
    """
    Returns:
        delta_states[key]

    PCA:
        delta_states[key] = delta_z [N, K]

    neuron:
        delta_states[key] = source_act - base_act [N, H]
    """
    delta_states = {}

    for key in base_states:
        basis = bases[key]
        base = base_states[key]
        source = source_states[key]

        chunks = []

        for start in range(0, len(base), batch_size):
            end = min(start + batch_size, len(base))

            diff = source[start:end].float() - base[start:end].float()

            if basis["mode"] == "pca":
                comps = basis["components"].float()  # [K, H]
                delta = diff @ comps.T              # [B, K]

            elif basis["mode"] == "neuron":
                delta = diff                        # [B, H]

            else:
                raise ValueError(f"Unknown basis mode: {basis['mode']}")

            chunks.append(delta.to(cache_dtype).cpu())

        delta_states[key] = torch.cat(chunks, dim=0)

    return delta_states


def make_patch_from_delta(
    base_act,
    delta,
    basis,
    start_dim,
    end_dim,
    strength=1.0,
):
    """
    base_act: [B, H]

    PCA:
        delta: [B, K] = delta_z

    neuron:
        delta: [B, H] = source_act - base_act

    Returns:
        x_patch: [B, H]
    """
    device = base_act.device

    base_act = base_act.float()
    delta = delta.to(device=device, dtype=torch.float32)

    if basis["mode"] == "pca":
        comps = basis["components"].to(device=device, dtype=torch.float32)

        if end_dim > comps.shape[0]:
            raise ValueError(
                f"Site end_dim={end_dim} exceeds PCA components={comps.shape[0]}"
            )

        delta_z = delta[:, start_dim:end_dim]          # [B, site_width]
        delta_x = delta_z @ comps[start_dim:end_dim]   # [B, H]

        return base_act + strength * delta_x

    elif basis["mode"] == "neuron":
        if end_dim > base_act.shape[1]:
            raise ValueError(
                f"Site end_dim={end_dim} exceeds hidden_size={base_act.shape[1]}"
            )

        patched = base_act.clone()
        patched[:, start_dim:end_dim] += strength * delta[:, start_dim:end_dim]
        return patched

    else:
        raise ValueError(f"Unknown basis mode: {basis['mode']}")



def precompute_base_positions(
    bank,
    sites,
    positions_are_padded=False,
):
    """
    Returns:
        padded_pos_by_token[token_id]: [N]
        logit_positions: [N]
    """
    base_mask = bank["base_attention_mask"].cpu()
    token_ids = sorted(set(token_id for _, token_id, _, _ in sites))

    padded_pos_by_token = {}

    for token_id in token_ids:
        pos = bank["base_position_by_id"][token_id].cpu()

        if positions_are_padded:
            padded = pos
        else:
            padded = (base_mask == 0).sum(dim=1) + pos

        padded_pos_by_token[token_id] = padded

    idx = torch.arange(base_mask.shape[1])
    logit_positions = (base_mask * idx.unsqueeze(0)).max(dim=1).values

    return padded_pos_by_token, logit_positions


@torch.no_grad()
def forward_sites_batch(
    model,
    base_input_ids,
    base_attention_mask,
    patch_positions_by_site,
    x_patches,
    sites,
    logit_positions,
    output_dtype=torch.float16,
    answer_label_ids=None,
):
    """
    Args:
        base_input_ids:          [B, T]
        base_attention_mask:     [B, T]
        patch_positions_by_site: [S, B]
        x_patches:               [S, B, H]
        sites:                   list[(layer_id, token_id, start, end)]
        logit_positions:         [B]

    Returns:
        logits_by_site: [S, B, num_answers] on CPU
    """
    device = next(model.parameters()).device

    base_input_ids = base_input_ids.to(device)
    base_attention_mask = base_attention_mask.to(device)
    patch_positions_by_site = patch_positions_by_site.to(device)
    x_patches = x_patches.to(device)
    logit_positions = logit_positions.to(device)

    S, B, H = x_patches.shape
    T = base_input_ids.shape[1]

    ids_rep = (
        base_input_ids
        .unsqueeze(0)
        .expand(S, B, T)
        .reshape(S * B, T)
    )

    mask_rep = (
        base_attention_mask
        .unsqueeze(0)
        .expand(S, B, T)
        .reshape(S * B, T)
    )

    position_ids = make_position_ids(base_attention_mask)
    position_ids_rep = (
        position_ids
        .unsqueeze(0)
        .expand(S, B, T)
        .reshape(S * B, T)
    )

    patch_pos_flat = patch_positions_by_site.reshape(S * B)
    patches_flat = x_patches.reshape(S * B, H)

    logit_pos_rep = (
        logit_positions
        .unsqueeze(0)
        .expand(S, B)
        .reshape(S * B)
    )

    rows_all = torch.arange(S * B, device=device)

    unique_layers = sorted(set(int(site[0]) for site in sites))
    handles = []

    def make_hook(layer_id):
        site_ids = [
            i for i, site in enumerate(sites)
            if int(site[0]) == int(layer_id)
        ]

        row_ids = torch.cat([
            torch.arange(i * B, (i + 1) * B, device=device)
            for i in site_ids
        ])

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            hidden = hidden.clone()

            hidden[row_ids, patch_pos_flat[row_ids], :] = patches_flat[row_ids].to(
                device=hidden.device,
                dtype=hidden.dtype,
            )

            if rest is not None:
                return (hidden,) + rest

            return hidden

        return hook

    for L in unique_layers:
        h = model.model.layers[int(L)].register_forward_hook(make_hook(L))
        handles.append(h)

    try:
        outputs = model.model(
            input_ids=ids_rep,
            attention_mask=mask_rep,
            position_ids=position_ids_rep,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for h in handles:
            h.remove()

    final_hidden = outputs.last_hidden_state
    hidden_at_pos = final_hidden[rows_all, logit_pos_rep, :]

    if answer_label_ids is None:
        logits = model.lm_head(hidden_at_pos)
    else:
        logits = answer_logits_from_hidden(
            model=model,
            hidden_at_pos=hidden_at_pos,
            answer_label_ids=answer_label_ids,
        )

    logits = logits.reshape(S, B, -1)

    return logits.detach().to(dtype=output_dtype, device="cpu")


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
    site_chunk_size=4,
    max_fit_states=4096,
    positions_are_padded=False,
    cache_dtype=torch.float16,
    output_dtype=torch.float16,
    return_bases=False,
):
    """
    Returns:
        {
            "sites": sites,
            "intervened_output": [num_sites, N * num_answers],
            "intervention_diff": [num_sites, N * num_answers],
        }

    sites:
        list[(layer_id, token_id, start_dim, end_dim)]
    """

    device = next(model.parameters()).device

    ### compute activations for base_input
    base_input_ids = bank["base_input_ids"]
    base_attention_mask = bank["base_attention_mask"]

    num_sites = len(sites)
    N = base_input_ids.shape[0]

    base_states, base_output = collect_site_activations(
        model=model,
        input_ids=base_input_ids,
        attention_mask=base_attention_mask,
        position_by_id=bank["base_position_by_id"],
        sites=sites,
        batch_size=batch_size,
        positions_are_padded=positions_are_padded,
        return_output=True,
        output_dtype=output_dtype,
        answer_label_ids=answer_label_ids,
    )
    # base_output: [N, answer_letter_size]

    num_answers = base_output.shape[-1]

    ### retrieve activations for all target_variables in source dataset
    source_states = collect_site_activations(
        model=model,
        input_ids=bank["source_input_ids"],
        attention_mask=bank["source_attention_mask"],
        position_by_id=bank["source_position_by_id"],
        sites=sites,
        batch_size=batch_size,
        positions_are_padded=positions_are_padded,
        return_output=False,
        answer_label_ids=answer_label_ids,
    )

    ### compute PCA directions (shared among causal variables)
    bases = fit_basis_for_sites(
        model=model,
        sites=sites,
        base_states=base_states,
        source_states=source_states,
        mode=mode,
        k=k,
        max_fit_states=max_fit_states,
    )

    ### compute delta for each target variable of source dataset
    delta_states = precompute_delta_states(
        base_states=base_states,
        source_states=source_states,
        bases=bases,
        cache_dtype=cache_dtype,
    )

    del source_states

    padded_pos_by_token, logit_positions_all = precompute_base_positions(
        bank=bank,
        sites=sites,
        positions_are_padded=positions_are_padded,
    )

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


    for site_start in range(0, num_sites, site_chunk_size):
        site_end = min(site_start + site_chunk_size, num_sites)
        sites_chunk = sites[site_start:site_end]
        S_chunk = len(sites_chunk)

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            B = batch_end - batch_start

            base_ids_b = base_input_ids[batch_start:batch_end]
            base_mask_b = base_attention_mask[batch_start:batch_end]
            logit_pos_b = logit_positions_all[batch_start:batch_end]

            patches = []
            patch_positions = []

            for L, token_id, a, b in sites_chunk:
                key = (int(L), token_id)

                x_base = base_states[key][batch_start:batch_end].to(device)
                delta = delta_states[key][batch_start:batch_end].to(device)

                x_patch = make_patch_from_delta(
                    base_act=x_base,
                    delta=delta,
                    basis=bases[key],
                    start_dim=a,
                    end_dim=b,
                    strength=strength,
                )

                patches.append(x_patch)

                pos = padded_pos_by_token[token_id][batch_start:batch_end]
                patch_positions.append(pos)

            x_patches = torch.stack(patches, dim=0)
            # [S_chunk, B, hidden]

            patch_positions_by_site = torch.stack(patch_positions, dim=0)
            # [S_chunk, B]

            after_b = forward_sites_batch(
                model=model,
                base_input_ids=base_ids_b,
                base_attention_mask=base_mask_b,
                patch_positions_by_site=patch_positions_by_site,
                x_patches=x_patches,
                sites=sites_chunk,
                logit_positions=logit_pos_b,
                output_dtype=output_dtype,
                answer_label_ids=answer_label_ids,
            )
            # after_b: [S_chunk, B, answer_letters_size] on CPU

            before_b = base_output[batch_start:batch_end]
            # before_b: [B, answer_letters_size]

            diff_b = after_b.float() - before_b.unsqueeze(0).float()
            diff_b = diff_b.to(output_dtype)

            # Flatten batch dimension into columns
            col_start = batch_start * num_answers
            col_end = batch_end * num_answers

            intervened_output[
                site_start:site_end,
                col_start:col_end,
            ] = after_b.reshape(S_chunk, B * num_answers)

            intervention_diff[
                site_start:site_end,
                col_start:col_end,
            ] = diff_b.reshape(S_chunk, B * num_answers)

            del x_patches
            del patch_positions_by_site
            del after_b
            del diff_b

    del delta_states
    del base_states
    del base_output

    result = {
        "sites": sites,
        "intervened_output": intervened_output,
        "intervention_diff": intervention_diff # [num_sites, N * num_answers]
    }

    if return_bases:
        result["bases"] = bases
    else:
        del bases


    return result


def get_topk_handle(T, sites, var_idx=0, top_k=1):

    row = torch.as_tensor(T[var_idx]).float().cpu()
    top_k = min(int(top_k), len(sites))
    values, indices = torch.topk(row, k=top_k)
    weights = values / values.sum()
    selected_sites = [sites[int(i)] for i in indices]

    return selected_sites, weights




def apply_soft_patch(
    current_act,
    source_act,
    basis,
    start_dim,
    end_dim,
    weight,
    strength,
):
    """
    current_act: [B, H]
        activation hiện tại trong forward pass.

    source_act: [B, H]
        activation source đã precompute.

    Returns:
        patched_act: [B, H]
    """
    current_act = current_act.float()
    source_act = source_act.to(
        device=current_act.device,
        dtype=torch.float32,
    )

    alpha = float(strength) * float(weight)

    if basis["mode"] == "neuron":
        patched = current_act.clone()

        patched[:, start_dim:end_dim] += alpha * (
            source_act[:, start_dim:end_dim]
            - current_act[:, start_dim:end_dim]
        )

        return patched

    if basis["mode"] == "pca":
        components = basis["components"].to(
            device=current_act.device,
            dtype=torch.float32,
        )
        # sklearn PCA format: [K, H]

        num_components = components.shape[0]

        if end_dim > num_components:
            raise ValueError(
                f"Site end_dim={end_dim} exceeds PCA components={num_components}"
            )

        diff = source_act - current_act
        # [B, H]

        delta_z = diff @ components.T
        # [B, K]

        delta_h = delta_z[:, start_dim:end_dim] @ components[start_dim:end_dim]
        # [B, H]

        return current_act + alpha * delta_h

    raise ValueError(f"Unknown basis mode: {basis['mode']}")


@torch.no_grad()
def forward_soft_intervene_batch(
    model,
    base_input_ids,
    base_attention_mask,
    selected_sites,
    weights,
    bases,
    source_states_b,
    padded_pos_by_token_b,
    logit_positions_b,
    answer_label_ids,
    strength=1.0,
    output_dtype=torch.float16,
):
    """
    Returns:
        logits: [B, num_answer_labels]
    """
    device = next(model.parameters()).device

    ids = base_input_ids.to(device)
    mask = base_attention_mask.to(device)

    B = ids.shape[0]
    rows = torch.arange(B, device=device)

    weights = torch.as_tensor(weights, device=device, dtype=torch.float32)
    logit_positions_b = logit_positions_b.to(device)

    source_states_b = {
        key: value.to(device)
        for key, value in source_states_b.items()
    }

    padded_pos_by_token_b = {
        token_id: pos.to(device)
        for token_id, pos in padded_pos_by_token_b.items()
    }

    layers = sorted(set(int(L) for L, _, _, _ in selected_sites))
    handles = []

    def make_hook(layer_id):
        sites_in_layer = [
            i for i, site in enumerate(selected_sites)
            if int(site[0]) == int(layer_id)
        ]

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            hidden = hidden.clone()

            for site_i in sites_in_layer:
                L, token_id, a, b = selected_sites[site_i]
                key = (int(L), token_id)

                pos = padded_pos_by_token_b[token_id]
                current_act = hidden[rows, pos, :]
                source_act = source_states_b[key]

                patched_act = apply_soft_patch(
                    current_act=current_act,
                    source_act=source_act,
                    basis=bases[key],
                    start_dim=int(a),
                    end_dim=int(b),
                    weight=weights[site_i],
                    strength=strength,
                )

                hidden[rows, pos, :] = patched_act.to(dtype=hidden.dtype)

            if rest is not None:
                return (hidden,) + rest

            return hidden

        return hook

    for L in layers:
        h = model.model.layers[int(L)].register_forward_hook(make_hook(L))
        handles.append(h)

    try:
        outputs = model.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=make_position_ids(mask),
            use_cache=False,
            return_dict=True,
        )
    finally:
        for h in handles:
            h.remove()

    final_hidden = outputs.last_hidden_state
    hidden_at_pos = final_hidden[rows, logit_positions_b, :]

    logits = answer_logits_from_hidden(
        model=model,
        hidden_at_pos=hidden_at_pos,
        answer_label_ids=answer_label_ids,
    )

    return logits.detach().to(dtype=output_dtype, device="cpu")




@torch.no_grad()
def evaluate_soft_intervention(
    model,
    bank,
    sites,
    T,
    bases,
    answer_label_ids,
    var_idx=0,
    var_name=None,
    top_k=1,
    strength=1.0,
    batch_size=16,
    positions_are_padded=False,
    output_dtype=torch.float16,
):
    """
    Returns:
        iia score: float
    """
    selected_sites, weights = get_topk_handle(
        T=T,
        sites=sites,
        var_idx=var_idx,
        top_k=top_k,
    )

    # Precompute source activations only for selected sites.
    source_states = collect_site_activations(
        model=model,
        input_ids=bank["source_input_ids"],
        attention_mask=bank["source_attention_mask"],
        position_by_id=bank["source_position_by_id"],
        sites=selected_sites,
        batch_size=batch_size,
        positions_are_padded=positions_are_padded,
        return_output=False,
    )

    padded_pos_by_token, logit_positions_all = precompute_base_positions(
        bank=bank,
        sites=selected_sites,
        positions_are_padded=positions_are_padded,
    )

    targets = torch.as_tensor(
        bank["counterfactual_label_ids"][var_name],
        dtype=torch.long,
    ).cpu()

    N = bank["base_input_ids"].shape[0]
    pred_chunks = []

    needed_keys = sorted(set(
        (int(L), token_id)
        for L, token_id, _, _ in selected_sites
    ))

    needed_tokens = sorted(set(
        token_id
        for _, token_id, _, _ in selected_sites
    ))

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        source_states_b = {
            key: source_states[key][start:end]
            for key in needed_keys
        }

        padded_pos_by_token_b = {
            token_id: padded_pos_by_token[token_id][start:end]
            for token_id in needed_tokens
        }

        logits_b = forward_soft_intervene_batch(
            model=model,
            base_input_ids=bank["base_input_ids"][start:end],
            base_attention_mask=bank["base_attention_mask"][start:end],
            selected_sites=selected_sites,
            weights=weights,
            bases=bases,
            source_states_b=source_states_b,
            padded_pos_by_token_b=padded_pos_by_token_b,
            logit_positions_b=logit_positions_all[start:end],
            answer_label_ids=answer_label_ids,
            strength=strength,
            output_dtype=output_dtype,
        )

        preds_b = logits_b.float().argmax(dim=-1)
        pred_chunks.append(preds_b.cpu())

    preds = torch.cat(pred_chunks, dim=0)

    iia = preds.eq(targets).float().mean().item()

    return iia



def calibrate_soft_intervention(
    model,
    cal_bank,
    sites,
    T,
    bases,
    answer_label_ids,
    top_k_values,
    strength_values,
    var_idx=0,
    var_name=None,
    batch_size=16,
    positions_are_padded=False,
    output_dtype=torch.float16,
):
    best_iia = -1.0
    best_top_k = None
    best_strength = None

    results = []

    for top_k in top_k_values:
        for strength in strength_values:
            iia = evaluate_soft_intervention(
                model=model,
                bank=cal_bank,
                sites=sites,
                T=T,
                bases=bases,
                answer_label_ids=answer_label_ids,
                var_idx=var_idx,
                var_name=var_name,
                top_k=top_k,
                strength=strength,
                batch_size=batch_size,
                positions_are_padded=positions_are_padded,
                output_dtype=output_dtype,
            )

            results.append({
                "top_k": int(top_k),
                "strength": float(strength),
                "iia": float(iia),
            })

            print(
                f"[CAL] top_k={top_k}, strength={strength}, IIA={iia:.4f}"
            )

            if iia > best_iia:
                best_iia = iia
                best_top_k = int(top_k)
                best_strength = float(strength)

    return {
        "best_iia": float(best_iia),
        "best_top_k": best_top_k,
        "best_strength": best_strength,
        "all_results": results,
    }


def normalize_rows(X, eps=1e-8):
    X = torch.as_tensor(X).float()
    return X / (X.norm(dim=1, keepdim=True) + eps)



@torch.no_grad()
def predict_bank_answer_labels(
    model,
    bank,
    input_prefix,
    answer_label_ids,
    batch_size=32,
):
    device = next(model.parameters()).device

    input_ids = bank[f"{input_prefix}_input_ids"]
    attention_mask = bank[f"{input_prefix}_attention_mask"]

    preds = []

    for start in range(0, input_ids.shape[0], batch_size):
        end = min(start + batch_size, input_ids.shape[0])

        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)

        rows = torch.arange(ids.shape[0], device=device)

        outputs = model.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=make_position_ids(mask),
            use_cache=False,
            return_dict=True,
        )

        logit_pos = last_real_token_positions(mask)
        hidden_at_pos = outputs.last_hidden_state[rows, logit_pos, :]

        logits = answer_logits_from_hidden(
            model=model,
            hidden_at_pos=hidden_at_pos,
            answer_label_ids=answer_label_ids,
        )

        preds.append(logits.float().argmax(dim=-1).cpu())

    return torch.cat(preds, dim=0)



@torch.no_grad()
def evaluate_oracle_full_layer_patches(
    model,
    bank,
    layers,
    token_id,
    answer_label_ids,
    batch_size=16,
    positions_are_padded=False,
    output_dtype=torch.float16,
):
    hidden_size = model.config.hidden_size

    sites = [
        (int(L), token_id, 0, hidden_size)
        for L in layers
    ]

    source_states = collect_site_activations(
        model=model,
        input_ids=bank["source_input_ids"],
        attention_mask=bank["source_attention_mask"],
        position_by_id=bank["source_position_by_id"],
        sites=sites,
        batch_size=batch_size,
        positions_are_padded=positions_are_padded,
        return_output=False,
    )

    padded_pos_by_token, logit_positions_all = precompute_base_positions(
        bank=bank,
        sites=sites,
        positions_are_padded=positions_are_padded,
    )

    targets = torch.as_tensor(
        bank["counterfactual_label_ids"],
        dtype=torch.long,
    ).cpu()

    base_targets = torch.as_tensor(
        bank["base_answer_label_ids"],
        dtype=torch.long,
    ).cpu()

    N = bank["base_input_ids"].shape[0]
    results = {}

    for L in layers:
        site = (int(L), token_id, 0, hidden_size)
        key = (int(L), token_id)

        pred_chunks = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            x_patches = source_states[key][start:end].unsqueeze(0)
            patch_positions = padded_pos_by_token[token_id][start:end].unsqueeze(0)

            logits_b = forward_sites_batch(
                model=model,
                base_input_ids=bank["base_input_ids"][start:end],
                base_attention_mask=bank["base_attention_mask"][start:end],
                patch_positions_by_site=patch_positions,
                x_patches=x_patches,
                sites=[site],
                logit_positions=logit_positions_all[start:end],
                output_dtype=output_dtype,
                answer_label_ids=answer_label_ids,
            )

            preds_b = logits_b[0].float().argmax(dim=-1)
            pred_chunks.append(preds_b.cpu())

        preds = torch.cat(pred_chunks, dim=0)

        iia = preds.eq(targets).float().mean().item()
        acc_base = preds.eq(base_targets).float().mean().item()
        changed = preds.ne(base_targets).float().mean().item()

        results[int(L)] = {
            "iia": float(iia),
            "acc_vs_base": float(acc_base),
            "changed_from_base": float(changed),
        }

        print(
            f"[oracle full patch] layer={int(L)} "
            f"IIA={iia:.4f} "
            f"acc_vs_base={acc_base:.4f} "
            f"changed={changed:.4f}"
        )

    return results





def run_plot_progressive(
    model,
    tokenizer,
    target_variable,
    layers,
    bands=None,
    band_count=8,
    mode_stage_A="neuron",
    mode_stage_B="pca",
    k=None,
    ft_size=128,
    cal_size=128,
    te_size=256,
    eps=4.0,
    top_k_values=range(1, 6),
    strength_values=(1, 2, 4, 8, 16, 32, 64),
    method="ot",
    chosen_token_position_id="correct_symbol",
    device="cuda",
):
    import string
    import torch

    answer_letters = tuple(string.ascii_uppercase)
    answer_label_ids = [
        letter_token_id(tokenizer, L)
        for L in answer_letters
    ]

    num_labels = len(answer_letters)
    hidden_size = model.config.hidden_size
    layers = list(layers)

    solver = {
        "ot": solve_ot,
        "uot": solve_uot,
    }[method]

    # ------------------------------------------------------------
    # 1. Build FT bank
    # ------------------------------------------------------------
    ft_bank = build_bank(
        model=model,
        tokenizer=tokenizer,
        target_variable=target_variable,
        bank_size=ft_size,
        answer_letters=answer_letters,
        device=device,
        example_offset=0,
    )

    # ------------------------------------------------------------
    # 2. Variable signature
    # ------------------------------------------------------------
    G, names = variable_signature(
        base_output=ft_bank["base_answer_label_ids"],
        dict_cf_outputs=ft_bank["counterfactual_label_ids"],
        num_labels=num_labels,
    )

    print("[G]", G.shape, names)

    G_solve = normalize_rows(G)

    # ------------------------------------------------------------
    # 3. Stage A: coarse layer localization, jointly over variables
    # ------------------------------------------------------------
    candidates_coarse_sites = [
        (L, chosen_token_position_id, 0, hidden_size)
        for L in layers
    ]

    sig_coarse = site_signature(
        model=model,
        bank=ft_bank,
        sites=candidates_coarse_sites,
        answer_label_ids=answer_label_ids,
        mode=mode_stage_A,
        strength=1.0,
        batch_size=32,
        site_chunk_size=32,
        output_dtype=torch.float16,
    )

    S_coarse = sig_coarse["intervention_diff"]
    S_coarse_solve = normalize_rows(S_coarse)

    T_coarse = solver(
        G_solve,
        S_coarse_solve,
        eps=eps,
    )

    print("[Stage A shapes]", G_solve.shape, S_coarse_solve.shape, T_coarse.shape)

    # T_coarse: [num_variables, num_layers]
    # Lấy top layers theo từng variable, rồi union lại thành candidate chung.
    num_candidate_layers = min(6, len(layers))

    layer_values, layer_indices = torch.topk(
        T_coarse,
        dim=1,
        k=num_candidate_layers,
        sorted=True,
    )

    print("[Stage A] layer_indices:")
    print(layer_indices)

    candidate_layers_by_variable = {}
    candidate_layer_masses_by_variable = {}

    for var_idx, var_name in enumerate(names):
        candidate_layers_var = [
            layers[int(i)]
            for i in layer_indices[var_idx].tolist()
        ]

        masses_var = layer_values[var_idx].tolist()

        candidate_layers_by_variable[var_name] = candidate_layers_var
        candidate_layer_masses_by_variable[var_name] = masses_var

        print(f"[Stage A] variable: {var_name}")
        print("[Stage A] candidate_layers:", candidate_layers_var)
        print("[Stage A] masses:", masses_var)

    candidate_layers = []

    for var_name in names:
        for layer in candidate_layers_by_variable[var_name]:
            if layer not in candidate_layers:
                candidate_layers.append(layer)

    print("[Stage A] candidate_layers union:", candidate_layers)

    del S_coarse
    del S_coarse_solve
    del sig_coarse

    # ------------------------------------------------------------
    # 4. Build CAL and TE banks
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 5. Stage B: fine localization jointly over variables
    # ------------------------------------------------------------
    best_candidate = None

    for candidate_layer in candidate_layers:
        print(f"\n[Stage B] trying layer {candidate_layer}")


        k_stage_B = k

        if mode_stage_B == "pca":
            if k_stage_B is None:
                k_stage_B = model.config.hidden_size

            if bands is None:
                bands_this_layer = make_pca_equal_bands_for_count(
                    pca_rank=k_stage_B,
                    band_count=band_count,
                )
            else:
                bands_this_layer = bands

        else:
            bands_this_layer = bands

            if bands_this_layer is None:
                raise ValueError("bands must be provided when mode_stage_B is not 'pca'.")

        candidates_fine_sites = [
            (candidate_layer, chosen_token_position_id, a, b)
            for (a, b) in bands_this_layer
        ]


        sig_fine = site_signature(
            model=model,
            bank=ft_bank,
            sites=candidates_fine_sites,
            answer_label_ids=answer_label_ids,
            mode=mode_stage_B,
            k=k_stage_B,
            strength=1.0,
            batch_size=32,
            site_chunk_size=32,
            output_dtype=torch.float16,
            return_bases=True,
        )

        S_fine = sig_fine["intervention_diff"]
        fine_bases = sig_fine["bases"]

        S_fine_solve = normalize_rows(S_fine)

        # Joint OT: all variables compete for the same fine sites.
        T_fine = solver(
            G_solve,
            S_fine_solve,
            eps=eps,
        )

        valid_top_k_values = [
            int(x)
            for x in top_k_values
            if int(x) <= len(candidates_fine_sites)
        ]

        if len(valid_top_k_values) == 0:
            valid_top_k_values = [len(candidates_fine_sites)]

        # --------------------------------------------------------
        # Joint calibration:
        # choose layer/top_k/strength by mean IIA over all variables.
        # --------------------------------------------------------
        best_layer_cal = None
        all_cal_results = []

        for top_k in valid_top_k_values:
            for strength in strength_values:
                iias = {}

                for var_idx, var_name in enumerate(names):
                    iia = evaluate_soft_intervention(
                        model=model,
                        bank=cal_bank,
                        sites=candidates_fine_sites,
                        T=T_fine,
                        bases=fine_bases,
                        answer_label_ids=answer_label_ids,
                        var_idx=var_idx,
                        var_name=var_name,
                        top_k=top_k,
                        strength=strength,
                        batch_size=16,
                        positions_are_padded=False,
                        output_dtype=torch.float16,
                    )

                    iias[var_name] = float(iia)

                mean_iia = sum(iias.values()) / len(iias)

                record = {
                    "layer": int(candidate_layer),
                    "top_k": int(top_k),
                    "strength": float(strength),
                    "mean_iia": float(mean_iia),
                    "iia_by_variable": iias,
                }

                all_cal_results.append(record)

                if best_layer_cal is None or mean_iia > best_layer_cal["mean_iia"]:
                    best_layer_cal = record

        print(
            f"[Layer {candidate_layer}] "
            f"best mean CAL IIA={best_layer_cal['mean_iia']:.4f}, "
            f"top_k={best_layer_cal['top_k']}, "
            f"strength={best_layer_cal['strength']}, "
            f"per_var={best_layer_cal['iia_by_variable']}"
        )

        candidate = {
            "layer": int(candidate_layer),
            "cal_iia": float(best_layer_cal["mean_iia"]),
            "cal_iia_by_variable": best_layer_cal["iia_by_variable"],
            "top_k": int(best_layer_cal["top_k"]),
            "strength": float(best_layer_cal["strength"]),
            "sites": candidates_fine_sites,
            "T_fine": T_fine,
            "bases": fine_bases,
            "cal_grid": all_cal_results,
        }

        if best_candidate is None or candidate["cal_iia"] > best_candidate["cal_iia"]:
            best_candidate = candidate

        del S_fine
        del S_fine_solve
        del sig_fine

    best_layer = best_candidate["layer"]
    best_top_k = best_candidate["top_k"]
    best_strength = best_candidate["strength"]

    print(
        f"\n[BEST JOINT] "
        f"layer={best_layer}, "
        f"mean CAL IIA={best_candidate['cal_iia']:.4f}, "
        f"top_k={best_top_k}, "
        f"strength={best_strength}, "
        f"per_var={best_candidate['cal_iia_by_variable']}"
    )

    # ------------------------------------------------------------
    # 6. Test each variable using the same joint-selected handle
    # ------------------------------------------------------------
    results = {}

    for var_idx, var_name in enumerate(names):
        test_iia = evaluate_soft_intervention(
            model=model,
            bank=te_bank,
            sites=best_candidate["sites"],
            T=best_candidate["T_fine"],
            bases=best_candidate["bases"],
            answer_label_ids=answer_label_ids,
            var_idx=var_idx,
            var_name=var_name,
            top_k=best_top_k,
            strength=best_strength,
            batch_size=16,
            positions_are_padded=False,
            output_dtype=torch.float16,
        )

        selected_sites, weights = get_topk_handle(
            T=best_candidate["T_fine"],
            sites=best_candidate["sites"],
            var_idx=var_idx,
            top_k=best_top_k,
        )

        results[var_name] = {
            "iia": float(test_iia),
            "cal_iia": float(best_candidate["cal_iia_by_variable"][var_name]),
            "joint_cal_iia": float(best_candidate["cal_iia"]),
            "best_layer": int(best_layer),
            "candidate_layers": candidate_layers,
            "candidate_layers_by_variable": candidate_layers_by_variable,
            "candidate_layer_masses_by_variable": candidate_layer_masses_by_variable,
            "top_k": int(best_top_k),
            "strength": float(best_strength),
            "selected_sites": selected_sites,
            "weights": weights.tolist(),
        }

    results["_joint"] = {
        "best_layer": int(best_layer),
        "joint_cal_iia": float(best_candidate["cal_iia"]),
        "cal_iia_by_variable": best_candidate["cal_iia_by_variable"],
        "top_k": int(best_top_k),
        "strength": float(best_strength),
        "candidate_layers": candidate_layers,
        "cal_grid": best_candidate["cal_grid"],
    }

    return results



if __name__ == "__main__":
    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device

    hidden_size = model.config.hidden_size

    target_variable = ("answer_pointer", "answer_token")
    # answer_letters = ["A", "B", "C", "D"]

    pca_k = 256

    pca_bands = make_pca_prefix_bands(
        ft_size=256,
        hidden_size=hidden_size,
        k=pca_k,
    )

    print("pca_bands:", pca_bands)


    result = run_plot_progressive(
        model=model,
        tokenizer=tokenizer,
        target_variable=target_variable,
        layers=list(range(model.config.num_hidden_layers)),
        bands=None,
        band_count=8,
        mode_stage_A="neuron",
        mode_stage_B="pca",
        k=pca_k,
        ft_size=256,
        cal_size=256,
        te_size=256,
        eps=1.0,
        top_k_values=(range(1, 129)),   # prefix bands overlap, test top_k=1 trước
        strength_values=(0.5, 1, 2, 4, 8, 16, 32, 64, 128),
        method="uot",
        chosen_token_position_id="last_token",
        device=device,
    )

    print(result)