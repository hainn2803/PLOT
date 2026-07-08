import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist


from mcqa_data import build_bank, letter_token_id
from mcqa_neural_net import load_gemma_model

from mcqa_ot import solve_ot, solve_uot

from sklearn.decomposition import PCA

import string

from mcqa_utils import normalize_rows, set_seed



def aggregate_family_signature(
    per_example_features,
    pair_source_families,
    family_order,
    normalize_blocks=True,
    eps=1e-8,
):
    """
    Aggregate per-example features by source family.

    Args:
        per_example_features:
            Tensor [N, D].

        pair_source_families:
            List[str] of length N.
            Example values:
                "answer_pointer"
                "answer_token"
                "both"

        family_order:
            Ordered tuple/list of family names.

        normalize_blocks:
            If True:
                1. center each family block
                2. L2-normalize each family block

    Returns:
        signature:
            Tensor [num_families * D]
    """
    X = torch.as_tensor(
        per_example_features,
        dtype=torch.float32,
    )

    if X.ndim != 2:
        raise ValueError(
            "per_example_features must have shape [N, D]"
        )

    if len(pair_source_families) != X.shape[0]:
        raise ValueError(
            "pair_source_families must have one entry "
            "per example"
        )

    family_blocks = []

    for family in family_order:
        mask_values = []

        for current_family in pair_source_families:
            mask_values.append(
                current_family == family
            )

        mask = torch.tensor(
            mask_values,
            dtype=torch.bool,
            device=X.device,
        )

        if bool(mask.any()):
            block = X[mask].mean(
                dim=0
            )

        else:
            block = torch.zeros(
                X.shape[1],
                dtype=X.dtype,
                device=X.device,
            )

        if normalize_blocks:
            # Center this family block
            block = (
                block
                - block.mean()
            )

            # L2 normalize this family block
            block_norm = torch.linalg.vector_norm(
                block,
                ord=2,
            )

            if float(block_norm.item()) > float(eps):
                block = (
                    block
                    / block_norm
                )

        family_blocks.append(
            block
        )

    signature = torch.cat(
        family_blocks,
        dim=0,
    )

    return signature



def variable_signature(
    base_output,
    dict_cf_outputs,
    pair_source_families,
    family_order,
    num_labels=None,
):
    """
    Build family-aggregated causal-variable signatures.

    Pipeline for each causal variable:
        1. One-hot base labels
        2. One-hot counterfactual labels
        3. Compute per-example label delta
        4. Mean within each source family
        5. Center each family block
        6. L2-normalize each family block
        7. Concatenate family blocks

    Returns:
        G_sig:
            [num_variables, num_families * num_labels]

        names:
            variable names
    """
    base_output = torch.as_tensor(
        base_output,
        dtype=torch.long,
    )

    base_onehot = F.one_hot(
        base_output,
        num_classes=num_labels,
    ).float()

    var_signatures = []
    names = []

    for var_name, cf_output in dict_cf_outputs.items():
        cf_output = torch.as_tensor(
            cf_output,
            dtype=torch.long,
        )

        cf_onehot = F.one_hot(
            cf_output,
            num_classes=num_labels,
        ).float()

        # [N, num_labels]
        delta = (
            cf_onehot
            - base_onehot
        )

        # [num_families * num_labels]
        signature = aggregate_family_signature(
            per_example_features=delta,
            pair_source_families=pair_source_families,
            family_order=family_order,
            normalize_blocks=True,
        )

        var_signatures.append(
            signature
        )

        names.append(
            var_name
        )

    G_sig = torch.stack(
        var_signatures,
        dim=0,
    )

    return G_sig, names



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

    site_weights = torch.as_tensor(
        site_weights,
        dtype=torch.float32,
        device=device,
    ).flatten()

    if site_weights.numel() != len(sites):
        raise ValueError(
            "site_weights must have one value per selected site"
        )

    if not torch.isfinite(site_weights).all():
        raise ValueError(
            "site_weights contains NaN or Inf"
        )

    weight_sum = site_weights.sum()

    if float(weight_sum.abs().item()) == 0.0:
        raise ValueError(
            "site_weights sum is zero; cannot normalize"
        )

    site_weights = site_weights / weight_sum

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

                    # site_weight = float(site_weights[site_idx])
                    site_weight = 1
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
    family_order=None,
):
    """
    Build family-aggregated neural-site effect signatures.

    For each candidate site:
        1. Run base forward pass
        2. Run intervention
        3. Compute per-example output delta
        4. Mean within each source family
        5. Center each family block
        6. L2-normalize each family block
        7. Concatenate family blocks

    Returns:
        intervention_diff:
            [num_sites, num_families * num_answers]
    """
    base_input_ids = bank[
        "base_input_ids"
    ]

    base_attention_mask = bank[
        "base_attention_mask"
    ]

    source_input_ids = bank[
        "source_input_ids"
    ]

    source_attention_mask = bank[
        "source_attention_mask"
    ]

    pair_source_families = bank[
        "pair_source_families"
    ]

    # --------------------------------------------------------
    # Family order
    # --------------------------------------------------------
    if family_order is None:
        family_order = bank[
            "source_families"
        ]

    family_order = tuple(
        family_order
    )

    num_sites = len(sites)
    N = base_input_ids.shape[0]

    # ========================================================
    # 1. Collect base activations and base outputs
    # ========================================================
    base_states, base_output = collect_site_activations(
        model=model,
        input_ids=base_input_ids,
        attention_mask=base_attention_mask,
        position_by_id=bank[
            "base_position_by_id"
        ],
        sites=sites,
        batch_size=batch_size,
        return_output=True,
        answer_label_ids=answer_label_ids,
        output_dtype=output_dtype,
    )

    # ========================================================
    # 2. Collect source activations
    # ========================================================
    source_states = collect_site_activations(
        model=model,
        input_ids=source_input_ids,
        attention_mask=source_attention_mask,
        position_by_id=bank[
            "source_position_by_id"
        ],
        sites=sites,
        batch_size=batch_size,
        return_output=False,
        answer_label_ids=answer_label_ids,
        output_dtype=output_dtype,
    )

    # ========================================================
    # 3. Build basis for each layer / token
    # ========================================================
    bases = precompute_basis_for_each_layer(
        model=model,
        sites=sites,
        base_states=base_states,
        source_states=source_states,
        mode=mode,
        k=k,
        max_fit_states=max_fit_states,
    )

    # ========================================================
    # 4. Allocate outputs
    # ========================================================
    num_answers = base_output.shape[-1]

    # Raw intervention outputs are still stored flattened.
    output_width = (
        N * num_answers
    )

    # Family-aggregated signature width.
    signature_width = (
        len(family_order)
        * num_answers
    )

    intervened_output = torch.empty(
        (num_sites, output_width),
        dtype=output_dtype,
        device="cpu",
    )

    intervention_diff = torch.empty(
        (num_sites, signature_width),
        dtype=output_dtype,
        device="cpu",
    )

    # ========================================================
    # 5. Build one effect signature per site
    # ========================================================
    for site_idx in range(num_sites):
        site = sites[
            site_idx
        ]

        patched_output = run_intervention(
            model=model,
            input_ids=base_input_ids,
            attention_mask=base_attention_mask,
            position_by_id=bank[
                "base_position_by_id"
            ],
            sites=[site],
            site_weights=[1.0],
            source_states=source_states,
            bases=bases,
            strength=strength,
            batch_size=batch_size,
            answer_label_ids=answer_label_ids,
            output_dtype=output_dtype,
        )

        # ----------------------------------------------------
        # Per-example neural effect
        #
        # Shape:
        #   [N, num_answers]
        # ----------------------------------------------------
        diff = (
            patched_output.float()
            - base_output.float()
        )

        # ----------------------------------------------------
        # Family-aggregated neural effect signature
        #
        # Shape:
        #   [num_families * num_answers]
        # ----------------------------------------------------
        site_effect_signature = (
            aggregate_family_signature(
                per_example_features=diff,
                pair_source_families=(
                    pair_source_families
                ),
                family_order=family_order,
                normalize_blocks=True,
            )
        )

        # Keep raw intervened outputs for debugging
        intervened_output[
            site_idx
        ] = (
            patched_output
            .reshape(-1)
            .to(output_dtype)
        )

        # Store family-aggregated site signature
        intervention_diff[site_idx] = (site_effect_signature.to(output_dtype))

    # ========================================================
    # 6. Return
    # ========================================================
    result = {
        "sites": sites,
        "base_output": base_output,
        "intervened_output": intervened_output,
        "intervention_diff": intervention_diff,
        "family_order": family_order,
    }

    if return_bases:
        result[
            "bases"
        ] = bases

    return result



def select_top_sites_from_T(
    T,
    sites,
    var_id,
    top_k,
    min_mass=1e-8,
):
    scores = T[var_id]

    valid_indices = []

    for site_idx in range(len(sites)):
        mass = scores[site_idx]

        if not bool(torch.isfinite(mass).item()):
            continue

        mass_value = float(
            mass.detach().cpu().item()
        )

        if mass_value <= float(min_mass):
            continue

        valid_indices.append(
            int(site_idx)
        )

    if len(valid_indices) == 0:
        raise ValueError(
            f"No finite OT-mass sites with mass >= {min_mass} "
            f"for var_id={var_id}"
        )

    top_k = min(
        int(top_k),
        len(valid_indices),
    )

    valid_scores = []

    for site_idx in valid_indices:
        valid_scores.append(
            scores[site_idx]
        )

    valid_scores = torch.stack(
        valid_scores,
        dim=0,
    )

    _, local_top_indices = torch.topk(
        valid_scores,
        k=top_k,
    )

    selected_sites = []
    selected_indices = []

    for local_idx in local_top_indices.tolist():
        site_idx = valid_indices[
            int(local_idx)
        ]

        selected_sites.append(
            sites[site_idx]
        )
        selected_indices.append(
            int(site_idx)
        )

    return selected_sites, selected_indices




def compute_iia(output_probs, target_labels):
    pred = output_probs.argmax(dim=-1).cpu()
    target = torch.as_tensor(target_labels, dtype=torch.long).cpu()

    return (pred == target).float().mean().item()


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
    stage_A_keep_layers=3,
    stage_A_iia_threshold=0.7,
    resolutions=(128, 144, 192, 256, 288, 384, 576, 768),
    top_k_values=range(1, 6),
    strength_values=(1, 2, 4, 8, 16, 32, 64),
    stage_A_strength_values=None,
    stage_A_method="ot",
    stage_B_method="ot",
    chosen_token_position_id="correct_symbol",
    device="cuda",
    seed=0,
):

    family_order = (
        "answer_pointer",
        "answer_token",
        "both",
    )

    set_seed(seed)

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
        base_output=ft_bank[
            "base_answer_label_ids"
        ],
        dict_cf_outputs=ft_bank[
            "counterfactual_label_ids"
        ],
        pair_source_families=ft_bank[
            "pair_source_families"
        ],
        family_order=family_order,
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
        family_order=family_order,
    )
    
    S_coarse_sig = sig_coarse["intervention_diff"]

    # T_coarse = stage_A_solver(
    #     normalize_rows(G_sig),
    #     normalize_rows(S_coarse_sig),
    #     eps=stage_A_eps,
    # )

    T_coarse = stage_A_solver(
        G_sig,
        S_coarse_sig,
        eps=stage_A_eps,
    )

    coarse_bases = sig_coarse["bases"]

    print("[Stage A shapes]", G_sig.shape, S_coarse_sig.shape, T_coarse.shape)


    # ============================================================
    # 4. Stage A selection + validation:
    #    1. Keep top stage_A_top_layers raw layers from T_coarse
    #    2. Validate each shortlisted layer on D_cal
    #    3. For each layer, keep its best strength / best cal IIA
    #    4. Remove layers with best cal IIA < stage_A_iia_threshold
    #    5. Sort remaining layers by best cal IIA
    #    6. Keep at most stage_A_keep_layers for Stage B
    #
    # Examples:
    #    multi-layer: top 6 -> threshold filter -> keep at most 3
    #    single-layer: top 6 -> threshold filter -> keep at most 1
    # ============================================================
    top_layers_var = {}
    top_coarse_by_var = {}
    stage_A_cal_results = []

    print(T_coarse)
    for var_id in range(len(names)):
        var_name = names[var_id]

        # --------------------------------------------------------
        # 1. Raw top-K layers from UOT coupling
        # --------------------------------------------------------
        top_sites, top_indices = select_top_sites_from_T(
            T=T_coarse,
            sites=coarse_sites,
            var_id=var_id,
            top_k=stage_A_top_layers,
            min_mass=0.0
        )

        calibrated_candidates = []

        print()
        print("[Stage A variable]", var_id, var_name)

        # --------------------------------------------------------
        # 2. Calibrate each shortlisted layer
        # --------------------------------------------------------
        for site_pos in range(len(top_sites)):
            site = top_sites[site_pos]
            site_index = int(top_indices[site_pos])

            L, token_id, start_dim, end_dim = site
            L = int(L)

            coupling_mass = float(
                T_coarse[var_id, site_index].detach().cpu()
            )

            print()
            print(
                "[Stage A raw candidate]",
                "var=", var_name,
                "rank=", site_pos + 1,
                "layer=", L,
                "mass=", coupling_mass,
            )

            # ----------------------------------------------------
            # Cache source activations for this layer
            # ----------------------------------------------------
            cal_source_states = collect_site_activations(
                model=model,
                input_ids=cal_bank["source_input_ids"],
                attention_mask=cal_bank["source_attention_mask"],
                position_by_id=cal_bank["source_position_by_id"],
                sites=[site],
                batch_size=32,
                return_output=False,
                answer_label_ids=answer_label_ids,
                output_dtype=torch.float16,
            )

            best_layer_candidate = None

            # ----------------------------------------------------
            # Sweep intervention strength
            # ----------------------------------------------------
            for strength in stage_A_strength_values:
                cal_output = run_intervention(
                    model=model,
                    input_ids=cal_bank["base_input_ids"],
                    attention_mask=cal_bank["base_attention_mask"],
                    position_by_id=cal_bank["base_position_by_id"],
                    sites=[site],
                    site_weights=[1.0],
                    source_states=cal_source_states,
                    bases=coarse_bases,
                    strength=float(strength),
                    batch_size=32,
                    answer_label_ids=answer_label_ids,
                    output_dtype=torch.float16,
                )

                cal_iia = compute_iia(
                    output_probs=cal_output,
                    target_labels=cal_bank[
                        "counterfactual_label_ids"
                    ][var_name],
                )

                candidate = {
                    "var_id": int(var_id),
                    "var_name": var_name,
                    "raw_rank": int(site_pos + 1),
                    "site_index": int(site_index),
                    "site": site,
                    "layer": int(L),
                    "coupling_mass": float(coupling_mass),
                    "strength": float(strength),
                    "cal_iia": float(cal_iia),
                }

                stage_A_cal_results.append(candidate)

                if best_layer_candidate is None:
                    best_layer_candidate = candidate
                else:
                    if candidate["cal_iia"] > best_layer_candidate["cal_iia"]:
                        best_layer_candidate = candidate

                print(
                    "[Stage A CAL]",
                    "var=", var_name,
                    "layer=", L,
                    "strength=", strength,
                    "iia=", round(float(cal_iia), 4),
                )

            # One best calibrated result per layer
            calibrated_candidates.append(best_layer_candidate)

        # --------------------------------------------------------
        # 3. Keep only layers that pass the validation threshold
        # --------------------------------------------------------
        threshold_candidates = []

        for candidate in calibrated_candidates:
            if (
                candidate["cal_iia"]
                >= float(stage_A_iia_threshold)
            ):
                threshold_candidates.append(
                    candidate
                )

        # --------------------------------------------------------
        # 4. Sort passing layers by best calibration IIA
        # --------------------------------------------------------
        threshold_candidates = sorted(
            threshold_candidates,
            key=lambda x: x["cal_iia"],
            reverse=True,
        )

        # --------------------------------------------------------
        # 5. Keep at most top-K passing layers
        # --------------------------------------------------------
        if int(stage_A_keep_layers) <= 0:
            raise ValueError(
                "stage_A_keep_layers must be at least 1"
            )

        if len(threshold_candidates) == 0:
            best_fallback = None

            for candidate in calibrated_candidates:
                if best_fallback is None:
                    best_fallback = candidate
                else:
                    if (
                        candidate["cal_iia"]
                        > best_fallback["cal_iia"]
                    ):
                        best_fallback = candidate

            selected_candidates = [
                best_fallback
            ]

            print(
                "[Stage A fallback]",
                "var=", var_name,
                "layer=", best_fallback["layer"],
                "cal_iia=", round(
                    best_fallback["cal_iia"],
                    4,
                ),
            )

        else:
            keep_count = min(
                int(stage_A_keep_layers),
                len(threshold_candidates),
            )

            selected_candidates = threshold_candidates[
                :keep_count
            ]

        selected_layers = []

        for candidate in selected_candidates:
            selected_layers.append(
                int(candidate["layer"])
            )

        top_layers_var[var_id] = selected_layers
        top_coarse_by_var[var_id] = selected_candidates

        # --------------------------------------------------------
        # Print final retained layers
        # --------------------------------------------------------
        print()
        print(
            "[Stage A retained layers]",
            "var=", var_name,
            "raw_top_k=", stage_A_top_layers,
            "iia_threshold=", stage_A_iia_threshold,
            "num_passing=", len(threshold_candidates),
            "keep_at_most=", stage_A_keep_layers,
            "num_retained=", len(selected_candidates),
        )

        for candidate in selected_candidates:
            print(
                "layer=", candidate["layer"],
                "raw_rank=", candidate["raw_rank"],
                "mass=", candidate["coupling_mass"],
                "best_strength=", candidate["strength"],
                "best_iia=", round(candidate["cal_iia"], 4),
            )


    print()
    print("[Stage A validated top_layers_var]")
    print(top_layers_var)




    # ============================================================
    # 5. Stage B: fine multi-layer top-k OT search
    # ============================================================
    best_by_var = {}
    stage_B_cal_results = []
    fine_cache = {}

    # Keep D_cal source activations out of fine_cache so they are
    # not returned/saved inside results.
    stage_B_cal_source_cache = {}

    for var_id in range(len(names)):
        var_name = names[var_id]
        stage_A_layers = top_layers_var[var_id]

        best_candidate = None

        print()
        print("[Stage B variable]", var_id, var_name)
        print("[Stage B Stage-A layers]", stage_A_layers)

        for resolution in resolutions:
            layer_key_list = []

            for L in stage_A_layers:
                layer_key_list.append(
                    int(L)
                )

            layer_key = tuple(
                layer_key_list
            )

            cache_key = (
                layer_key,
                int(resolution),
            )

            # ----------------------------------------------------
            # Build fine sites, signatures, and OT once per
            # (stage_A_layers, resolution)
            # ----------------------------------------------------
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
                    batch_size=32,
                    output_dtype=torch.float16,
                    return_bases=True,
                    family_order=family_order,
                )

                S_fine_sig = (
                    sig_fine["intervention_diff"]
                )

                T_fine = stage_B_solver(
                    G_sig,
                    S_fine_sig,
                    eps=stage_B_eps,
                )

                # T_fine = stage_B_solver(
                #     normalize_rows(G_sig),
                #     normalize_rows(S_fine_sig),
                #     eps=stage_B_eps,
                # )

                fine_cache[cache_key] = {
                    "stage_A_layers": layer_key,
                    "sites_fine": sites_fine,
                    "T_fine": T_fine,
                    "S_fine": S_fine_sig,
                    "bases": sig_fine["bases"],
                }

            cached = fine_cache[cache_key]

            sites_fine = cached["sites_fine"]
            T_fine = cached["T_fine"]
            bases_fine = cached["bases"]

            # ----------------------------------------------------
            # Cache D_cal source activations once per fine family.
            # collect_site_activations stores full vectors per
            # (layer, token), so the same cache works for every
            # top-k subset from these sites.
            # ----------------------------------------------------
            if cache_key not in stage_B_cal_source_cache:
                cal_source_states = collect_site_activations(
                    model=model,
                    input_ids=cal_bank["source_input_ids"],
                    attention_mask=cal_bank["source_attention_mask"],
                    position_by_id=cal_bank["source_position_by_id"],
                    sites=sites_fine,
                    batch_size=32,
                    return_output=False,
                    answer_label_ids=answer_label_ids,
                    output_dtype=torch.float16,
                )

                stage_B_cal_source_cache[cache_key] = (
                    cal_source_states
                )

            cal_source_states = (
                stage_B_cal_source_cache[cache_key]
            )

            # ----------------------------------------------------
            # Non-greedy selection:
            # for each top_k, take the top-k sites directly from
            # the OT coupling row T_fine[var_id].
            # ----------------------------------------------------
            for top_k in top_k_values:
                selected_sites, selected_indices = (
                    select_top_sites_from_T(
                        T=T_fine,
                        sites=sites_fine,
                        var_id=var_id,
                        top_k=top_k,
                    )
                )

                selected_weights = T_fine[
                    var_id,
                    selected_indices,
                ]

                selected_weights = (
                    selected_weights
                    .detach()
                    .float()
                    .cpu()
                )

                for strength in strength_values:
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

                    cal_iia = compute_iia(
                        output_probs=cal_output,
                        target_labels=cal_bank[
                            "counterfactual_label_ids"
                        ][var_name],
                    )

                    candidate = {
                        "var_id": int(var_id),
                        "var_name": var_name,
                        "stage_A_layers": layer_key,
                        "resolution": int(resolution),
                        "top_k": int(len(selected_sites)),
                        "requested_top_k": int(top_k),
                        "strength": float(strength),
                        "cal_iia": float(cal_iia),
                        "selected_indices": selected_indices,
                        "selected_sites": selected_sites,
                        "selected_weights": selected_weights,
                        "cache_key": cache_key,
                        "selection_method": "top_k_ot",
                    }

                    stage_B_cal_results.append(
                        candidate
                    )

                    if best_candidate is None:
                        best_candidate = candidate
                    else:
                        if (
                            candidate["cal_iia"]
                            > best_candidate["cal_iia"]
                        ):
                            best_candidate = candidate

                    print(
                        "[Stage B CAL]",
                        "var=", var_name,
                        "layers=", layer_key,
                        "resolution=", resolution,
                        "top_k=", len(selected_sites),
                        "strength=", strength,
                        "iia=", round(
                            candidate["cal_iia"],
                            4,
                        ),
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
        selected_weights = best_candidate["selected_weights"]
        strength = best_candidate["strength"]

        cache_key = (
            tuple(stage_A_layers),
            int(resolution),
        )

        bases_fine = fine_cache[cache_key]["bases"]

        # --------------------------------------------------------
        # Collect source activations from all selected sites
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
        # Coupling-weighted multi-layer intervention
        # --------------------------------------------------------
        te_output = run_intervention(
            model=model,
            input_ids=te_bank["base_input_ids"],
            attention_mask=te_bank["base_attention_mask"],
            position_by_id=te_bank["base_position_by_id"],
            sites=selected_sites,
            site_weights=selected_weights,
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
            "selected_weights=", selected_weights,
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

        ft_size=200,
        cal_size=100,
        te_size=100,

        stage_A_eps=0.003,
        stage_B_eps=0.003,
        stage_A_method="uot",
        stage_B_method="ot",

        # Stage A:
        # 1) shortlist top 6 layers by OT/UOT mass
        # 2) validate all 6 on D_cal
        # 3) discard layers with cal_iia < threshold
        # 4) keep at most top 3 passing layers
        #
        # For single-layer Stage B, set stage_A_keep_layers=1.
        stage_A_top_layers=6,
        stage_A_keep_layers=1,
        stage_A_iia_threshold=0.7,

        resolutions = (128, 144, 192, 256, 288, 384, 576, 768),
        top_k_values = (1, 2, 4),
        strength_values = (0.5, 1, 2, 4, 8, 16, 32, 64),

        chosen_token_position_id="last_token",
        device=device,
        seed=1
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
            "stage_A_layers=", r["stage_A_layers"],
            "resolution=", r["resolution"],
            "top_k=", r["top_k"],
            "strength=", r["strength"],
            "cal_iia=", r["cal_iia"],
            "test_iia=", r["test_iia"],
        )
