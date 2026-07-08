import torch
import os
import re
from datasets import load_dataset
from mcqa_neural_net import load_gemma_model
from mcqa_constants import DATASET_PATH, HF_TOKEN
from functools import lru_cache


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_TOKEN"] = HF_TOKEN

def choice_labels_from_prompt(prompt):
    labels = re.findall(r"(?m)^\s*([A-Z])\.\s+", prompt)

    if not labels:
        raise ValueError(f"Could not parse choice labels from prompt:\n{prompt}")

    return labels


def find_answer_relative_position(prompt, answer_letter):
    labels = choice_labels_from_prompt(prompt)

    if answer_letter not in labels:
        raise ValueError(
            f"answer_letter={answer_letter!r} not found in labels={labels}"
        )

    return labels.index(answer_letter)


 
def find_answer_letter(prompt, choices):
    """Correct answer letter for a prompt.
 
    Args:
        prompt:  str, the MCQA prompt; the queried color appears as 'X is/are <color>.'.
        choices: dict with 'text' (list[str] colors) and 'label' (list[str] letters),
                 aligned by index.
    Returns:
        str: the label of the choice whose color matches the queried color.
    """
    queried_color = re.search(r"\b(?:is|are)\s+(\w+)\.", prompt).group(1)
    position = choices["text"].index(queried_color)
    return choices["label"][position]
 
 
def load_mcqa_dataset(number_of_examples, split="train"):
    """Load and parse the first `number_of_examples` rows.
 
    Args:
        number_of_examples: int, how many rows to take from the split.
        split:              str, dataset split name (default 'train').
    Returns:
        list[dict], each with base_prompt, base_answer_letter, and the pointer/
        token counterfactual prompts with their correct answer letters.
    """
    dataset = load_dataset(DATASET_PATH, split=split, cache_dir=CACHE_DIR)
    take = min(int(number_of_examples), len(dataset))
    dataset = dataset.select(range(take))
 
    examples = []
    for row in dataset:
        pointer_cf = row["answerPosition_counterfactual"]
        token_cf = row["randomLetter_counterfactual"]
        both_cf = row["answerPosition_randomLetter_counterfactual"]
        examples.append({

            "base_prompt": row["prompt"],
            "base_answer_letter": find_answer_letter(row["prompt"], row["choices"]),

            "source_prompt_pointer": pointer_cf["prompt"],
            "source_answer_letter_pointer": find_answer_letter(pointer_cf["prompt"], pointer_cf["choices"]),

            "source_prompt_token": token_cf["prompt"],
            "source_answer_letter_token": find_answer_letter(token_cf["prompt"], token_cf["choices"]),

            "source_prompt_both": both_cf["prompt"],
            "source_answer_letter_both": find_answer_letter(both_cf["prompt"], both_cf["choices"]),

        })
    return examples
 

@lru_cache(maxsize=None)
def letter_token_id(tokenizer, letter):
    """Single-token id of an answer letter. Cached: each distinct letter is
    tokenized only once.
 
    Args:
        tokenizer: a HF tokenizer (hashable; used as part of the cache key).
        letter:    str, a single answer letter (e.g. 'A', 'W').
    Returns:
        int: the token id of ' letter' (or 'letter'), whichever is one token.
    """
    for candidate in (" " + letter, letter):
        ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"letter {letter!r} is not a single token")
 
 
def find_index_of_letter_position_in_prompt(tokenizer, prompt, letter):
    """Token index (unpadded) of the answer letter inside the prompt.
 
    The answer always appears as 'X. <color>', so locate 'X.', tokenize the
    prefix up to and including X, and take the last token.
 
    Args:
        tokenizer: a HF tokenizer.
        prompt:    str, the full prompt.
        letter:    str, the answer letter to locate.
    Returns:
        int: index of the letter's token in the un-padded token sequence.
    """
    char_index = prompt.index(letter + ".")
    prefix_ids = tokenizer(prompt[: char_index + 1], add_special_tokens=True)["input_ids"]
    return len(prefix_ids) - 1
 
 
@torch.no_grad()
def predict_and_encode(model, tokenizer, prompts, gold_letters, device):
    """Tokenize one batch, predict the next token, and return both.
 
    Args:
        model, tokenizer: loaded Gemma-2 LM and its tokenizer (left-padding).
        prompts:          list[str], one batch.
        gold_letters:     list[str], correct letter per prompt.
        device:           'cuda' or 'cpu'.
    Returns:
        (correct, input_ids, attention_mask):
            correct        list[bool], argmax == gold token.
            input_ids      [batch, seq] (cpu) the batch's token ids.
            attention_mask [batch, seq] (cpu) 1 = real, 0 = pad.
    """
    encoding = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    predicted_ids = model(**encoding, logits_to_keep=1).logits.squeeze(1).argmax(dim=-1)
    gold_ids = torch.tensor([letter_token_id(tokenizer, l) for l in gold_letters], device=device)
    correct = (predicted_ids == gold_ids).cpu().tolist()
    return correct, encoding["input_ids"].cpu(), encoding["attention_mask"].cpu()
 
 
def _left_pad_sequences(sequences, pad_id):
    """Left-pad a list of 1-D token tensors to a common width.
 
    Args:
        sequences: list of 1-D LongTensors (real tokens, no padding).
        pad_id:    int, pad token id.
    Returns:
        (input_ids, attention_mask): [n, width] tensors; real tokens sit on the
        right so the last column (-1) is always a real token.
    """
    width = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for i, seq in enumerate(sequences):
        input_ids[i, width - len(seq):] = seq
        attention_mask[i, width - len(seq):] = 1
    return input_ids, attention_mask
 

def select_correct_pairs(model, tokenizer, base_prompts, base_letters,
                         source_prompts, source_letters, bank_size, device, batch_size=32):
    """Find pairs Gemma answers correctly on BOTH sides, EARLY-STOPPING, and reuse
    the filter's tokenization so kept prompts are never tokenized twice.
 
    Processes the pool batch by batch; for each kept row the already-computed
    token sequence (stripped to real tokens) is saved, then all kept rows are
    left-padded once to a common width. Stops as soon as bank_size are found.
 
    Args:
        model, tokenizer:               loaded Gemma-2 LM and its tokenizer.
        base_prompts, base_letters:     list[str], the base side of the pool.
        source_prompts, source_letters: list[str], the source side (aligned).
        bank_size:                      int, number of good pairs wanted.
        device:                         'cuda' or 'cpu'.
        batch_size:                     int, pairs evaluated per forward.
    Returns:
        (kept, base_ids, base_mask, source_ids, source_mask):
            kept       list[int], kept pool indices (len <= bank_size).
            base_ids   [n, seq] base token ids (left-padded), n = len(kept).
            base_mask  [n, seq] base attention mask.
            source_ids/source_mask: same for the source side.
    """
    pad_id = tokenizer.pad_token_id
    kept, base_seqs, source_seqs = [], [], []
    pool_size = len(base_prompts)
    for start in range(0, pool_size, batch_size):
        idx = list(range(start, min(start + batch_size, pool_size)))
        base_ok, base_ids, base_mask = predict_and_encode(
            model, tokenizer, [base_prompts[i] for i in idx], [base_letters[i] for i in idx], device)
        source_ok, source_ids, source_mask = predict_and_encode(
            model, tokenizer, [source_prompts[i] for i in idx], [source_letters[i] for i in idx], device)
        for j, i in enumerate(idx):
            if base_ok[j] and source_ok[j]:
                kept.append(i)
                base_seqs.append(base_ids[j][base_mask[j] == 1])         # strip to real tokens
                source_seqs.append(source_ids[j][source_mask[j] == 1])
        if len(kept) >= bank_size:
            print(f"kept {bank_size} pairs after scanning {start + len(idx)}/{pool_size} examples (early stop)")
            break
    else:
        print(f"pool exhausted: only {len(kept)}/{bank_size} pairs found; increase oversample")
 
    kept = kept[:bank_size]
    base_seqs, source_seqs = base_seqs[:bank_size], source_seqs[:bank_size]
    base_ids, base_mask = _left_pad_sequences(base_seqs, pad_id)         # re-pad all kept once
    source_ids, source_mask = _left_pad_sequences(source_seqs, pad_id)
    return kept, base_ids, base_mask, source_ids, source_mask


def _add_pair(ex, source_prompt_key, source_answer_key):
    base_prompt = ex["base_prompt"]
    base_answer_letter = ex["base_answer_letter"]

    source_prompt = ex[source_prompt_key]
    source_answer_letter = ex[source_answer_key]

    # AP intervention:
    # use source answer position, but read the letter from base choices.
    source_relpos = find_answer_relative_position(source_prompt, source_answer_letter)
    base_choice_labels = choice_labels_from_prompt(base_prompt)
    cf_answer_pointer = base_choice_labels[source_relpos]

    # AT intervention:
    # use source answer token directly.
    cf_answer_token = source_answer_letter

    return base_prompt, base_answer_letter, source_prompt, source_answer_letter, cf_answer_pointer, cf_answer_token


def _normalize_target_variable_name(target_variable):
    tv = str(target_variable).lower()

    if tv in {"ap", "answer_pointer", "answerposition"}:
        return "answer_pointer"

    if tv in {"at", "answer_token", "randomletter"}:
        return "answer_token"

    raise ValueError(
        f"Unknown target_variable={target_variable!r}. "
        "Use AP/answer_pointer or AT/answer_token."
    )


def _slice_bank(bank, start, end):
    """Slice a build_bank() result without changing its public structure."""
    sliced = {}

    tensor_row_keys = {
        "base_input_ids",
        "base_attention_mask",
        "source_input_ids",
        "source_attention_mask",
        "base_answer_label_ids",
        "base_answer_pointer_ids",
        "source_answer_pointer_ids",
    }

    list_row_keys = {
        "pair_source_families",
    }

    for key, value in bank.items():
        if key in tensor_row_keys:
            sliced[key] = value[start:end]

        elif key in list_row_keys:
            sliced[key] = value[start:end]

        elif key == "counterfactual_label_ids":
            sliced[key] = {}

            for var_name, labels in value.items():
                sliced[key][var_name] = labels[start:end]

        elif key == "changed_mask":
            sliced[key] = {}

            for var_name, mask in value.items():
                sliced[key][var_name] = mask[start:end]

        elif key in {"base_position_by_id", "source_position_by_id"}:
            sliced[key] = {}

            for position_id, positions in value.items():
                sliced[key][position_id] = positions[start:end]

        else:
            # Metadata / shared tensors such as label_space.
            sliced[key] = value

    return sliced


def build_bank(
    model,
    tokenizer,
    target_variable=("AT", "AP"),
    bank_size=256,
    answer_letters=None,
    device="cuda",
    example_offset=0,
    oversample=2,
    filter_batch_size=32,
    source_families=None,
    shuffle_pairs=True,
    seed=0,
    sensitivity_target=None,
):
    """
    Build one MCQA pair bank.

    Backward compatible with the previous build_bank() API.

    New optional argument:
        sensitivity_target:
            None              -> shared/unrestricted bank (fit/train behavior)
            "answer_pointer"  -> keep only rows where AP changes
            "answer_token"    -> keep only rows where AT changes

    This lets calibration/test banks be target-specific while old scripts that
    do not pass sensitivity_target continue to run unchanged.
    """
    import string
    import random
    import torch

    # ------------------------------------------------------------
    # 1. Normalize target variables
    # ------------------------------------------------------------
    if isinstance(target_variable, str):
        target_variable = [target_variable]

    target_variables = []

    for tv in target_variable:
        normalized_tv = _normalize_target_variable_name(tv)

        if normalized_tv not in target_variables:
            target_variables.append(normalized_tv)

    target_variables = tuple(target_variables)

    # ------------------------------------------------------------
    # 2. Normalize source families
    # ------------------------------------------------------------
    if source_families is None:
        if set(target_variables) == {"answer_pointer", "answer_token"}:
            source_families = (
                "answer_pointer",
                "answer_token",
                "both",
            )
        else:
            source_families = target_variables

    elif isinstance(source_families, str):
        source_families = (source_families,)

    normalized_source_families = []

    for sf in source_families:
        sf = sf.lower()

        if sf in {"ap", "answer_pointer", "answerposition"}:
            normalized_source_families.append("answer_pointer")

        elif sf in {"at", "answer_token", "randomletter"}:
            normalized_source_families.append("answer_token")

        elif sf in {"both", "ap_at", "answer_pointer_answer_token"}:
            normalized_source_families.append("both")

        else:
            raise ValueError(f"Unknown source_family={sf!r}")

    source_families = tuple(dict.fromkeys(normalized_source_families))

    # ------------------------------------------------------------
    # 3. Normalize sensitivity target
    # ------------------------------------------------------------
    if sensitivity_target is not None:
        sensitivity_target = _normalize_target_variable_name(
            sensitivity_target
        )

        if sensitivity_target not in target_variables:
            raise ValueError(
                "sensitivity_target must be included in target_variable"
            )

    # ------------------------------------------------------------
    # 4. Output label space
    # ------------------------------------------------------------
    if answer_letters is None:
        if "answer_token" in target_variables:
            answer_letters = tuple(string.ascii_uppercase)
        else:
            answer_letters = tuple("ABCD")
    else:
        answer_letters = tuple(answer_letters)

    label_ids = []

    for letter in answer_letters:
        label_ids.append(
            letter_token_id(tokenizer, letter)
        )

    label_space = torch.tensor(
        label_ids,
        dtype=torch.long,
    )

    letter_to_label = {}

    for i, letter in enumerate(answer_letters):
        letter_to_label[letter] = i

    # ------------------------------------------------------------
    # 5. Load a candidate pool
    # ------------------------------------------------------------
    # Sensitive holdout banks need a much larger pool because many rows are
    # discarded when the target variable does not change.
    effective_oversample = int(oversample)

    if sensitivity_target is not None:
        effective_oversample = max(
            effective_oversample,
            20,
        )

    number_of_examples = (
        int(example_offset)
        + int(bank_size) * effective_oversample
    )

    examples = load_mcqa_dataset(
        number_of_examples=number_of_examples,
    )

    examples = examples[int(example_offset):]

    # ------------------------------------------------------------
    # 6. Flatten examples into base-source pairs
    # ------------------------------------------------------------
    pair_base_prompts = []
    pair_base_letters = []
    pair_base_pointer_ids = []

    pair_source_prompts = []
    pair_source_letters = []
    pair_source_pointer_ids = []

    pair_source_families = []

    pair_cf_values = {}

    for tv in target_variables:
        pair_cf_values[tv] = []

    pair_changed = {
        "answer_pointer": [],
        "answer_token": [],
    }

    def append_pair(
        ex,
        source_family,
        source_prompt_key,
        source_answer_key,
    ):
        base_prompt = ex["base_prompt"]
        base_answer_letter = ex["base_answer_letter"]

        source_prompt = ex[source_prompt_key]
        source_answer_letter = ex[source_answer_key]

        base_pointer = find_answer_relative_position(
            base_prompt,
            base_answer_letter,
        )

        source_pointer = find_answer_relative_position(
            source_prompt,
            source_answer_letter,
        )

        pair_base_prompts.append(base_prompt)
        pair_base_letters.append(base_answer_letter)
        pair_base_pointer_ids.append(int(base_pointer))

        pair_source_prompts.append(source_prompt)
        pair_source_letters.append(source_answer_letter)
        pair_source_pointer_ids.append(int(source_pointer))

        pair_source_families.append(source_family)

        # Repo semantics:
        # AP label is source answer_pointer (0..3).
        # AT label is source answer symbol in alphabet space (0..25).
        for tv in target_variables:
            if tv == "answer_pointer":
                pair_cf_values[tv].append(
                    int(source_pointer)
                )

            elif tv == "answer_token":
                pair_cf_values[tv].append(
                    int(letter_to_label[source_answer_letter])
                )

        pair_changed["answer_pointer"].append(
            int(base_pointer) != int(source_pointer)
        )

        pair_changed["answer_token"].append(
            str(base_answer_letter) != str(source_answer_letter)
        )

    for ex in examples:
        if "answer_pointer" in source_families:
            append_pair(
                ex=ex,
                source_family="answer_pointer",
                source_prompt_key="source_prompt_pointer",
                source_answer_key="source_answer_letter_pointer",
            )

        if "answer_token" in source_families:
            append_pair(
                ex=ex,
                source_family="answer_token",
                source_prompt_key="source_prompt_token",
                source_answer_key="source_answer_letter_token",
            )

        if "both" in source_families:
            append_pair(
                ex=ex,
                source_family="both",
                source_prompt_key="source_prompt_both",
                source_answer_key="source_answer_letter_both",
            )

    # ------------------------------------------------------------
    # 7. Target-specific sensitivity filtering for holdout banks
    # ------------------------------------------------------------
    if sensitivity_target is not None:
        sensitive_indices = []

        target_changed = pair_changed[
            sensitivity_target
        ]

        for i in range(len(target_changed)):
            if bool(target_changed[i]):
                sensitive_indices.append(i)

        def keep_indices(values, indices):
            kept_values = []

            for i in indices:
                kept_values.append(values[i])

            return kept_values

        pair_base_prompts = keep_indices(
            pair_base_prompts,
            sensitive_indices,
        )
        pair_base_letters = keep_indices(
            pair_base_letters,
            sensitive_indices,
        )
        pair_base_pointer_ids = keep_indices(
            pair_base_pointer_ids,
            sensitive_indices,
        )

        pair_source_prompts = keep_indices(
            pair_source_prompts,
            sensitive_indices,
        )
        pair_source_letters = keep_indices(
            pair_source_letters,
            sensitive_indices,
        )
        pair_source_pointer_ids = keep_indices(
            pair_source_pointer_ids,
            sensitive_indices,
        )

        pair_source_families = keep_indices(
            pair_source_families,
            sensitive_indices,
        )

        for tv in target_variables:
            pair_cf_values[tv] = keep_indices(
                pair_cf_values[tv],
                sensitive_indices,
            )

        for tv in pair_changed:
            pair_changed[tv] = keep_indices(
                pair_changed[tv],
                sensitive_indices,
            )

        print(
            "[build_bank sensitivity]",
            "target=", sensitivity_target,
            "candidates=", len(pair_base_prompts),
        )

    # ------------------------------------------------------------
    # 8. Shuffle before factual filtering
    # ------------------------------------------------------------
    if shuffle_pairs:
        if sensitivity_target is None:
            rng = random.Random(int(seed))
        else:
            # Target-specific deterministic shuffle, matching the repo's
            # separate AP/AT holdout randomization idea.
            rng = random.Random(
                f"{int(seed)}:holdout:{sensitivity_target}"
            )

        order = []

        for i in range(len(pair_base_prompts)):
            order.append(i)

        rng.shuffle(order)

        def reorder(values):
            reordered = []

            for i in order:
                reordered.append(values[i])

            return reordered

        pair_base_prompts = reorder(pair_base_prompts)
        pair_base_letters = reorder(pair_base_letters)
        pair_base_pointer_ids = reorder(pair_base_pointer_ids)

        pair_source_prompts = reorder(pair_source_prompts)
        pair_source_letters = reorder(pair_source_letters)
        pair_source_pointer_ids = reorder(pair_source_pointer_ids)

        pair_source_families = reorder(pair_source_families)

        for tv in target_variables:
            pair_cf_values[tv] = reorder(
                pair_cf_values[tv]
            )

        for tv in pair_changed:
            pair_changed[tv] = reorder(
                pair_changed[tv]
            )

    # ------------------------------------------------------------
    # 9. Factual filtering: model correct on BOTH base and source
    # ------------------------------------------------------------
    (
        kept,
        base_input_ids,
        base_attention_mask,
        source_input_ids,
        source_attention_mask,
    ) = select_correct_pairs(
        model=model,
        tokenizer=tokenizer,
        base_prompts=pair_base_prompts,
        base_letters=pair_base_letters,
        source_prompts=pair_source_prompts,
        source_letters=pair_source_letters,
        bank_size=bank_size,
        device=device,
        batch_size=filter_batch_size,
    )

    if len(kept) < int(bank_size):
        raise ValueError(
            f"Could only build {len(kept)}/{bank_size} rows "
            f"for sensitivity_target={sensitivity_target!r}. "
            "Increase oversample or enlarge the raw dataset pool."
        )

    def pick(values):
        picked = []

        for i in kept:
            picked.append(values[i])

        return picked

    # ------------------------------------------------------------
    # 10. Labels
    # ------------------------------------------------------------
    picked_base_letters = pick(pair_base_letters)

    base_label_values = []

    for letter in picked_base_letters:
        base_label_values.append(
            int(letter_to_label[letter])
        )

    base_answer_label_ids = torch.tensor(
        base_label_values,
        dtype=torch.long,
    )

    counterfactual_label_ids = {}

    for tv in target_variables:
        counterfactual_label_ids[tv] = torch.tensor(
            pick(pair_cf_values[tv]),
            dtype=torch.long,
        )

    picked_base_pointer_ids = torch.tensor(
        pick(pair_base_pointer_ids),
        dtype=torch.long,
    )

    picked_source_pointer_ids = torch.tensor(
        pick(pair_source_pointer_ids),
        dtype=torch.long,
    )

    changed_mask = {}

    for tv in target_variables:
        changed_mask[tv] = torch.tensor(
            pick(pair_changed[tv]),
            dtype=torch.bool,
        )

    # ------------------------------------------------------------
    # 11. Token positions for intervention sites
    # ------------------------------------------------------------
    picked_base_positions = []
    picked_source_positions = []

    for i in kept:
        base_pos = find_index_of_letter_position_in_prompt(
            tokenizer,
            pair_base_prompts[i],
            pair_base_letters[i],
        )

        source_pos = find_index_of_letter_position_in_prompt(
            tokenizer,
            pair_source_prompts[i],
            pair_source_letters[i],
        )

        picked_base_positions.append(base_pos)
        picked_source_positions.append(source_pos)

    base_positions = torch.tensor(
        picked_base_positions,
        dtype=torch.long,
    )

    source_positions = torch.tensor(
        picked_source_positions,
        dtype=torch.long,
    )

    base_last_token_positions = (
        base_attention_mask.sum(dim=1) - 1
    )

    source_last_token_positions = (
        source_attention_mask.sum(dim=1) - 1
    )

    picked_source_families = pick(
        pair_source_families
    )

    # ------------------------------------------------------------
    # 12. Return bank (old keys preserved, new diagnostics added)
    # ------------------------------------------------------------
    return {
        "target_variables": target_variables,
        "source_families": source_families,
        "pair_source_families": picked_source_families,
        "sensitivity_target": sensitivity_target,

        "label_space": label_space,

        "base_input_ids": base_input_ids,
        "base_attention_mask": base_attention_mask,

        "source_input_ids": source_input_ids,
        "source_attention_mask": source_attention_mask,

        "base_position_by_id": {
            "correct_symbol": base_positions,
            "correct_symbol_period": base_positions + 1,
            "last_token": base_last_token_positions,
        },

        "source_position_by_id": {
            "correct_symbol": source_positions,
            "correct_symbol_period": source_positions + 1,
            "last_token": source_last_token_positions,
        },

        "base_answer_label_ids": base_answer_label_ids,
        "counterfactual_label_ids": counterfactual_label_ids,

        # Extra repo-like diagnostics / semantics.
        "base_answer_pointer_ids": picked_base_pointer_ids,
        "source_answer_pointer_ids": picked_source_pointer_ids,
        "changed_mask": changed_mask,
    }


def build_target_specific_cal_test_banks(
    model,
    tokenizer,
    target_variable=("answer_pointer", "answer_token"),
    cal_size=100,
    te_size=100,
    answer_letters=None,
    device="cuda",
    example_offset=0,
    oversample=20,
    filter_batch_size=32,
    source_families=None,
    seed=0,
):
    """
    Build separate target-sensitive calibration/test banks.

    Returns:
        cal_banks["answer_pointer"]
        cal_banks["answer_token"]
        te_banks["answer_pointer"]
        te_banks["answer_token"]

    For each target variable independently:
        1. keep only rows where that target changes
        2. use target-specific deterministic shuffle
        3. factual-filter base/source correctness
        4. take first cal_size rows for calibration
        5. take next te_size rows for test
    """
    if isinstance(target_variable, str):
        target_variable = (target_variable,)

    target_variables = []

    for tv in target_variable:
        normalized_tv = _normalize_target_variable_name(tv)

        if normalized_tv not in target_variables:
            target_variables.append(normalized_tv)

    target_variables = tuple(target_variables)

    combined_size = int(cal_size) + int(te_size)

    cal_banks = {}
    te_banks = {}

    for target_name in target_variables:
        combined_bank = build_bank(
            model=model,
            tokenizer=tokenizer,
            target_variable=target_variables,
            bank_size=combined_size,
            answer_letters=answer_letters,
            device=device,
            example_offset=example_offset,
            oversample=oversample,
            filter_batch_size=filter_batch_size,
            source_families=source_families,
            shuffle_pairs=True,
            seed=seed,
            sensitivity_target=target_name,
        )

        cal_banks[target_name] = _slice_bank(
            combined_bank,
            0,
            int(cal_size),
        )

        te_banks[target_name] = _slice_bank(
            combined_bank,
            int(cal_size),
            combined_size,
        )

        print(
            "[target-specific banks]",
            "target=", target_name,
            "cal_size=", cal_banks[target_name]["base_input_ids"].shape[0],
            "test_size=", te_banks[target_name]["base_input_ids"].shape[0],
            "cal_changed=", int(
                cal_banks[target_name]["changed_mask"][target_name]
                .sum()
                .item()
            ),
            "test_changed=", int(
                te_banks[target_name]["changed_mask"][target_name]
                .sum()
                .item()
            ),
        )

    return cal_banks, te_banks

 
if __name__ == "__main__":
    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device

    dataset = load_mcqa_dataset(number_of_examples=10, split="train")

    print(dataset[0])

    # bank = build_bank(
    #     model=model,
    #     tokenizer=tokenizer,
    #     target_variable=["AT", "AP"],
    #     bank_size=64,
    #     device=device,
    #     oversample=4,
    # )

    # print(bank["target_variables"])
    # print(bank["answer_letters"])
    # print(bank["base_input_ids"].shape)

    # print(bank["targets"]["answer_pointer"]["source_input_ids"].shape)
    # print(bank["targets"]["answer_token"]["source_input_ids"].shape)

    # print(torch.unique(bank["targets"]["answer_pointer"]["counterfactual_label_ids"]))
    # print(torch.unique(bank["targets"]["answer_token"]["counterfactual_label_ids"]))
