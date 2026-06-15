import torch
import os
import re
from datasets import load_dataset
from mcqa_neural_net import load_gemma_model
from mcqa_constants import DATASET_PATH
from functools import lru_cache


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 

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
    dataset = dataset.select(range(number_of_examples))
 
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



def build_bank_single_target_variable(model, tokenizer, target_variable, bank_size,
               answer_letters=("A", "B", "C", "D"), device="cuda",
               example_offset=0, oversample=2, filter_batch_size=32):
    """Build a filtered, tokenized bank of (base, source) pairs for one variable.
 
    Steps: load -> keep pairs Gemma answers correctly on BOTH sides -> tokenize
    kept prompts -> record answer-token positions and labels.
 
    Args:
        model, tokenizer:  loaded Gemma-2 LM and its tokenizer.
        target_variable:   'answer_pointer' or 'answer_token'; selects which
                           counterfactual column is the source.
        bank_size:         int, number of pairs to keep.
        answer_letters:    tuple[str], the label space (defines label ids 0..L-1).
        device:            'cuda' or 'cpu'.
        example_offset:    int, skip this many rows first (for disjoint ft/cal/te banks).
        oversample:        int, load oversample*bank_size rows to absorb filtering.
        filter_batch_size: int, batch size for the correctness filter.
    Returns:
        dict of parallel tensors/lists (entry i is one pair):
            base/source_input_ids, base/source_attention_mask : [bank, seq] tokenized prompts
            base/source_answer_positions                       : [bank] unpadded answer-token index
            base/source_position_by_id                         : {'correct_symbol': the above}
            base_answer_token_ids, counterfactual_token_ids    : [bank] real token ids (for IIA argmax)
            base_answer_label_ids, counterfactual_label_ids    : [bank] label ids 0..L-1 (for signatures)
            label_space                                        : [L] answer-letter token ids
            counterfactual_letters                             : list[str] source answer letters
    """
    source_prompt_key, source_letter_key = {
        "answer_pointer": ("source_prompt_pointer", "source_answer_letter_pointer"),
        "answer_token": ("source_prompt_token", "source_prompt_token"),
    }[target_variable]
 
    # 1) load text (oversample to absorb the filter), skipping example_offset rows
    examples = load_mcqa_dataset(example_offset + bank_size * oversample)[example_offset:]
    base_prompts = [ex["base_prompt"] for ex in examples]
    base_letters = [ex["base_answer_letter"] for ex in examples]
    source_prompts = [ex[source_prompt_key] for ex in examples]
    source_letters = [ex[source_letter_key] for ex in examples]
 
    # 2) keep pairs Gemma gets right on BOTH sides (early stop) AND reuse the
    #    filter's tokenization -> kept prompts are tokenized only once
    kept, base_input_ids, base_attention_mask, source_input_ids, source_attention_mask = (
        select_correct_pairs(model, tokenizer, base_prompts, base_letters,
                             source_prompts, source_letters, bank_size, device, filter_batch_size))
    pick = lambda values: [values[i] for i in kept]
 
    # 3) answer-token positions + answer ids (token ids cached via letter_token_id)
    base_positions = torch.tensor(
        [find_index_of_letter_position_in_prompt(tokenizer, base_prompts[i], base_letters[i]) for i in kept])
    source_positions = torch.tensor(
        [find_index_of_letter_position_in_prompt(tokenizer, source_prompts[i], source_letters[i]) for i in kept])
    base_answer_ids = [letter_token_id(tokenizer, base_letters[i]) for i in kept]
    counterfactual_letters = pick(source_letters)
    counterfactual_ids = [letter_token_id(tokenizer, l) for l in counterfactual_letters]

    # ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    # position_last = len(ids) - 1
 
    # 4) token id -> label id 0..L-1 (label_space = answer_letters)
    label_space = [letter_token_id(tokenizer, L) for L in answer_letters]
    token_to_label = {tid: i for i, tid in enumerate(label_space)}
    base_answer_label_ids = [token_to_label[t] for t in base_answer_ids]
    counterfactual_label_ids = [token_to_label[t] for t in counterfactual_ids]
 
    return {
        "base_input_ids": base_input_ids,
        "base_attention_mask": base_attention_mask,
        "source_input_ids": source_input_ids,
        "source_attention_mask": source_attention_mask,
        "base_answer_positions": base_positions,
        "source_answer_positions": source_positions,
        "base_position_by_id": {
            "correct_symbol": base_positions,
            "correct_symbol_period": base_positions + 1,
            "last_token": base_attention_mask.sum(1) - 1,
            },
        "source_position_by_id": {
            "correct_symbol": source_positions, 
            "correct_symbol_period": source_positions + 1,
            "last_token": source_attention_mask.sum(1) - 1,
            },
        "base_answer_token_ids": torch.tensor(base_answer_ids),
        "counterfactual_token_ids": torch.tensor(counterfactual_ids),
        "base_answer_label_ids": torch.tensor(base_answer_label_ids),
        "counterfactual_label_ids": torch.tensor(counterfactual_label_ids),
        "label_space": torch.tensor(label_space),
        "counterfactual_letters": counterfactual_letters,
    }



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
):
    import string
    import random

    # ------------------------------------------------------------
    # 1. Normalize target variables
    # ------------------------------------------------------------
    if isinstance(target_variable, str):
        target_variable = [target_variable]

    target_variables = []

    for tv in target_variable:
        tv = tv.lower()

        if tv in {"ap", "answer_pointer", "answerposition"}:
            target_variables.append("answer_pointer")

        elif tv in {"at", "answer_token", "randomletter"}:
            target_variables.append("answer_token")

        else:
            raise ValueError(
                f"Unknown target_variable={tv!r}. "
                "Use AP/answer_pointer or AT/answer_token. "
                "'both' is a source family, not a causal variable."
            )

    target_variables = tuple(dict.fromkeys(target_variables))

    # ------------------------------------------------------------
    # 2. Normalize source families
    # ------------------------------------------------------------
    if source_families is None:
        if set(target_variables) == {"answer_pointer", "answer_token"}:
            source_families = ("answer_pointer", "answer_token", "both")
        else:
            source_families = target_variables

    elif isinstance(source_families, str):
        source_families = (source_families,)

    normalized_families = []

    for fam in source_families:
        fam = fam.lower()

        if fam in {"ap", "answer_pointer", "answerposition"}:
            normalized_families.append("answer_pointer")

        elif fam in {"at", "answer_token", "randomletter"}:
            normalized_families.append("answer_token")

        elif fam in {"both", "answerposition_randomletter"}:
            normalized_families.append("both")

        else:
            raise ValueError(f"Unknown source family: {fam!r}")

    source_families = tuple(dict.fromkeys(normalized_families))

    # ------------------------------------------------------------
    # 3. Output label space
    # ------------------------------------------------------------
    if answer_letters is None:
        if "answer_token" in target_variables:
            answer_letters = tuple(string.ascii_uppercase)
        else:
            answer_letters = tuple("ABCD")
    else:
        answer_letters = tuple(answer_letters)

    label_space = torch.tensor(
        [letter_token_id(tokenizer, L) for L in answer_letters],
        dtype=torch.long,
    )

    letter_to_label = {
        letter: i
        for i, letter in enumerate(answer_letters)
    }

    # ------------------------------------------------------------
    # 4. Load raw examples
    # ------------------------------------------------------------
    examples = load_mcqa_dataset(
        example_offset + bank_size * oversample
    )[example_offset:]

    # ------------------------------------------------------------
    # 5. Flatten examples into base-source pairs
    # ------------------------------------------------------------
    pair_base_prompts = []
    pair_base_letters = []

    pair_source_prompts = []
    pair_source_letters = []

    pair_cf_letters = {
        tv: []
        for tv in target_variables
    }

    def add_pair(ex, source_prompt_key, source_answer_key):
        base_prompt = ex["base_prompt"]
        base_answer_letter = ex["base_answer_letter"]

        source_prompt = ex[source_prompt_key]
        source_answer_letter = ex[source_answer_key]

        # Needed only here to build AP counterfactual output.
        source_relpos = find_answer_relative_position(
            source_prompt,
            source_answer_letter,
        )

        base_choice_labels = choice_labels_from_prompt(base_prompt)

        if source_relpos >= len(base_choice_labels):
            raise ValueError(
                f"source_relpos={source_relpos} out of range for "
                f"base_choice_labels={base_choice_labels}"
            )

        # AP intervention:
        # use source answer position, but read the letter from base choices.
        cf_answer_pointer = base_choice_labels[source_relpos]

        # AT intervention:
        # use source answer token directly.
        cf_answer_token = source_answer_letter

        pair_base_prompts.append(base_prompt)
        pair_base_letters.append(base_answer_letter)

        pair_source_prompts.append(source_prompt)
        pair_source_letters.append(source_answer_letter)

        for tv in target_variables:
            if tv == "answer_pointer":
                pair_cf_letters[tv].append(cf_answer_pointer)

            elif tv == "answer_token":
                pair_cf_letters[tv].append(cf_answer_token)

    for ex in examples:
        if "answer_pointer" in source_families:
            add_pair(
                ex,
                source_prompt_key="source_prompt_pointer",
                source_answer_key="source_answer_letter_pointer",
            )

        if "answer_token" in source_families:
            add_pair(
                ex,
                source_prompt_key="source_prompt_token",
                source_answer_key="source_answer_letter_token",
            )

        if "both" in source_families:
            add_pair(
                ex,
                source_prompt_key="source_prompt_both",
                source_answer_key="source_answer_letter_both",
            )

    # ------------------------------------------------------------
    # 6. Shuffle flattened pairs before filtering
    # ------------------------------------------------------------
    if shuffle_pairs:
        rng = random.Random(seed)
        order = list(range(len(pair_base_prompts)))
        rng.shuffle(order)

        pair_base_prompts = [pair_base_prompts[i] for i in order]
        pair_base_letters = [pair_base_letters[i] for i in order]

        pair_source_prompts = [pair_source_prompts[i] for i in order]
        pair_source_letters = [pair_source_letters[i] for i in order]

        for tv in target_variables:
            pair_cf_letters[tv] = [
                pair_cf_letters[tv][i]
                for i in order
            ]

    # ------------------------------------------------------------
    # 7. Keep only pairs where model predicts base and source correctly
    # ------------------------------------------------------------
    kept, base_input_ids, base_attention_mask, source_input_ids, source_attention_mask = (
        select_correct_pairs(
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
    )

    def pick(values):
        return [values[i] for i in kept]

    # ------------------------------------------------------------
    # 8. Convert base/counterfactual letters to label ids
    # ------------------------------------------------------------
    picked_base_letters = pick(pair_base_letters)

    base_answer_label_ids = torch.tensor(
        [letter_to_label[letter] for letter in picked_base_letters],
        dtype=torch.long,
    )

    counterfactual_label_ids = {}

    for tv in target_variables:
        picked_cf_letters = pick(pair_cf_letters[tv])

        counterfactual_label_ids[tv] = torch.tensor(
            [letter_to_label[letter] for letter in picked_cf_letters],
            dtype=torch.long,
        )

    # ------------------------------------------------------------
    # 9. Token positions for intervention sites
    # ------------------------------------------------------------
    base_positions = torch.tensor([
        find_index_of_letter_position_in_prompt(
            tokenizer,
            pair_base_prompts[i],
            pair_base_letters[i],
        )
        for i in kept
    ], dtype=torch.long)

    source_positions = torch.tensor([
        find_index_of_letter_position_in_prompt(
            tokenizer,
            pair_source_prompts[i],
            pair_source_letters[i],
        )
        for i in kept
    ], dtype=torch.long)

    # ------------------------------------------------------------
    # 10. Minimal bank needed for signature/evaluate
    # ------------------------------------------------------------
    return {
        "target_variables": target_variables,

        # Used by site_signature/evaluate to read only answer logits.
        "label_space": label_space,

        # Used by site_signature/evaluate.
        "base_input_ids": base_input_ids,
        "base_attention_mask": base_attention_mask,
        "source_input_ids": source_input_ids,
        "source_attention_mask": source_attention_mask,

        # Used by intervention code to know where to patch/read.
        "base_position_by_id": {
            "correct_symbol": base_positions,
            "correct_symbol_period": base_positions + 1,
            "last_token": base_attention_mask.sum(1) - 1,
        },

        "source_position_by_id": {
            "correct_symbol": source_positions,
            "correct_symbol_period": source_positions + 1,
            "last_token": source_attention_mask.sum(1) - 1,
        },

        # Used by variable_signature.
        "base_answer_label_ids": base_answer_label_ids,
        "counterfactual_label_ids": counterfactual_label_ids,
    }



 
if __name__ == "__main__":
    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device

    bank = build_bank(
        model=model,
        tokenizer=tokenizer,
        target_variable=["AT", "AP"],
        bank_size=64,
        device=device,
        oversample=4,
    )

    print(bank["target_variables"])
    print(bank["answer_letters"])
    print(bank["base_input_ids"].shape)

    print(bank["targets"]["answer_pointer"]["source_input_ids"].shape)
    print(bank["targets"]["answer_token"]["source_input_ids"].shape)

    print(torch.unique(bank["targets"]["answer_pointer"]["counterfactual_label_ids"]))
    print(torch.unique(bank["targets"]["answer_token"]["counterfactual_label_ids"]))
