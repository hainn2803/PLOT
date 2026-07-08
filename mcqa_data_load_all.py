import os
import random
import re
import string
from functools import lru_cache

import torch
from datasets import load_dataset

from mcqa_constants import DATASET_PATH, HF_TOKEN
from mcqa_neural_net import load_gemma_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_TOKEN"] = HF_TOKEN

TARGETS = ("answer_pointer", "answer_token")
FAMILIES = {
    "answer_pointer": "answerPosition_counterfactual",
    "answer_token": "randomLetter_counterfactual",
    "both": "answerPosition_randomLetter_counterfactual",
}
ANSWER_LETTERS = tuple(string.ascii_uppercase)


def choice_labels_from_prompt(prompt):
    """Return the ordered answer labels shown in an MCQA prompt."""
    labels = re.findall(r"(?m)^\s*([A-Z])\.\s+", prompt)
    if not labels:
        raise ValueError(f"Could not parse choice labels from prompt:\n{prompt}")
    return labels


def find_answer_relative_position(prompt, answer_letter):
    """Return the answer pointer index of one answer letter."""
    labels = choice_labels_from_prompt(prompt)
    if answer_letter not in labels:
        raise ValueError(f"answer_letter={answer_letter!r} not found in labels={labels}")
    return labels.index(answer_letter)


def find_answer_letter(prompt, choices):
    """Return the correct answer letter from the queried color and choices."""
    match = re.search(r"\b(?:is|are)\s+(\w+)\.", prompt)
    if match is None:
        raise ValueError(f"Could not parse queried color from prompt:\n{prompt}")
    return choices["label"][choices["text"].index(match.group(1))]


@lru_cache(maxsize=None)
def letter_token_id(tokenizer, letter):
    """Return a single-token id for one answer letter."""
    for text in (" " + letter, letter):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"letter {letter!r} is not a single token")


def find_index_of_letter_position_in_prompt(tokenizer, prompt, letter):
    """Return the unpadded token index of one choice letter in the prompt."""
    match = re.search(rf"(?m)^\s*{re.escape(letter)}\.", prompt)
    if match is None:
        raise ValueError(f"Could not find choice letter {letter!r} in prompt:\n{prompt}")
    char_index = match.end() - 2
    ids = tokenizer(prompt[:char_index + 1], add_special_tokens=True)["input_ids"]
    return len(ids) - 1


def load_mcqa_pairs(dataset_size=None, split="train"):
    """Load raw rows and expand each row into the three counterfactual families."""
    dataset = load_dataset(DATASET_PATH, split=split, cache_dir=CACHE_DIR)
    if dataset_size is not None:
        dataset = dataset.select(range(min(int(dataset_size), len(dataset))))

    pairs_by_family = {}
    for family in FAMILIES:
        pairs_by_family[family] = []

    for row in dataset:
        base_prompt = row["prompt"]
        base_letter = find_answer_letter(base_prompt, row["choices"])
        base_pointer = find_answer_relative_position(base_prompt, base_letter)

        for family, key in FAMILIES.items():
            source = row[key]
            source_prompt = source["prompt"]
            source_letter = find_answer_letter(source_prompt, source["choices"])
            source_pointer = find_answer_relative_position(source_prompt, source_letter)
            pairs_by_family[family].append({
                "base_prompt": base_prompt,
                "base_answer_letter": base_letter,
                "base_answer_pointer": base_pointer,
                "source_prompt": source_prompt,
                "source_answer_letter": source_letter,
                "source_answer_pointer": source_pointer,
                "source_family": family,
            })
    return pairs_by_family


@torch.no_grad()
def predict_correct(model, tokenizer, prompts, gold_letters, device, batch_size=32):
    """Return whether Gemma predicts the correct next-token answer for every prompt."""
    correct = []
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        encoding = tokenizer(prompts[start:end], padding=True, return_tensors="pt").to(device)
        predicted_ids = model(**encoding, logits_to_keep=1).logits[:, -1].argmax(dim=-1)
        gold_ids = []
        for letter in gold_letters[start:end]:
            gold_ids.append(letter_token_id(tokenizer, letter))
        gold_ids = torch.tensor(gold_ids, device=device)
        correct.extend((predicted_ids == gold_ids).cpu().tolist())
    return correct


def factual_filter(model, tokenizer, pairs_by_family, device, batch_size=32):
    """Keep rows where Gemma answers both base and source prompts correctly, family by family."""
    filtered = {}
    for family, pairs in pairs_by_family.items():
        base_prompts, base_letters, source_prompts, source_letters = [], [], [], []
        for pair in pairs:
            base_prompts.append(pair["base_prompt"])
            base_letters.append(pair["base_answer_letter"])
            source_prompts.append(pair["source_prompt"])
            source_letters.append(pair["source_answer_letter"])

        base_ok = predict_correct(model, tokenizer, base_prompts, base_letters, device, batch_size)
        source_ok = predict_correct(model, tokenizer, source_prompts, source_letters, device, batch_size)
        kept = []
        for i, pair in enumerate(pairs):
            if base_ok[i] and source_ok[i]:
                kept.append(pair)
        filtered[family] = kept
        print(f"[filter] family={family} kept={len(kept)}/{len(pairs)}")
    return filtered


def build_bank(tokenizer, rows):
    """Tokenize selected rows and build the tensor bank used by intervention code."""
    if not rows:
        raise ValueError("Cannot build an empty bank")

    base_prompts, source_prompts = [], []
    base_letters, source_letters = [], []
    base_pointers, source_pointers = [], []
    source_families = []
    for row in rows:
        base_prompts.append(row["base_prompt"])
        source_prompts.append(row["source_prompt"])
        base_letters.append(row["base_answer_letter"])
        source_letters.append(row["source_answer_letter"])
        base_pointers.append(row["base_answer_pointer"])
        source_pointers.append(row["source_answer_pointer"])
        source_families.append(row["source_family"])

    base_encoding = tokenizer(base_prompts, padding=True, return_tensors="pt")
    source_encoding = tokenizer(source_prompts, padding=True, return_tensors="pt")

    base_positions, source_positions = [], []
    for i in range(len(rows)):
        base_positions.append(find_index_of_letter_position_in_prompt(tokenizer, base_prompts[i], base_letters[i]))
        source_positions.append(find_index_of_letter_position_in_prompt(tokenizer, source_prompts[i], source_letters[i]))

    letter_to_label = {}
    label_space = []
    for i, letter in enumerate(ANSWER_LETTERS):
        letter_to_label[letter] = i
        label_space.append(letter_token_id(tokenizer, letter))

    base_labels, pointer_labels, token_labels = [], [], []
    changed_pointer, changed_token = [], []
    for i in range(len(rows)):
        base_labels.append(letter_to_label[base_letters[i]])
        pointer_labels.append(source_pointers[i])
        token_labels.append(letter_to_label[source_letters[i]])
        changed_pointer.append(base_pointers[i] != source_pointers[i])
        changed_token.append(base_letters[i] != source_letters[i])

    base_positions = torch.tensor(base_positions, dtype=torch.long)
    source_positions = torch.tensor(source_positions, dtype=torch.long)
    base_mask = base_encoding["attention_mask"].to(torch.long)
    source_mask = source_encoding["attention_mask"].to(torch.long)

    return {
        "target_variables": TARGETS,
        "source_families": tuple(FAMILIES),
        "pair_source_families": source_families,
        "label_space": torch.tensor(label_space, dtype=torch.long),
        "base_input_ids": base_encoding["input_ids"].to(torch.long),
        "base_attention_mask": base_mask,
        "source_input_ids": source_encoding["input_ids"].to(torch.long),
        "source_attention_mask": source_mask,
        "base_position_by_id": {
            "correct_symbol": base_positions,
            "correct_symbol_period": base_positions + 1,
            "last_token": base_mask.sum(dim=1) - 1,
        },
        "source_position_by_id": {
            "correct_symbol": source_positions,
            "correct_symbol_period": source_positions + 1,
            "last_token": source_mask.sum(dim=1) - 1,
        },
        "base_answer_label_ids": torch.tensor(base_labels, dtype=torch.long),
        "counterfactual_label_ids": {
            "answer_pointer": torch.tensor(pointer_labels, dtype=torch.long),
            "answer_token": torch.tensor(token_labels, dtype=torch.long),
        },
        "base_answer_pointer_ids": torch.tensor(base_pointers, dtype=torch.long),
        "source_answer_pointer_ids": torch.tensor(source_pointers, dtype=torch.long),
        "changed_mask": {
            "answer_pointer": torch.tensor(changed_pointer, dtype=torch.bool),
            "answer_token": torch.tensor(changed_token, dtype=torch.bool),
        },
    }


def build_mcqa_banks(model, tokenizer, train_pool_size=128, cal_size=128, te_size=256, dataset_size=None, split="train", device="cuda", batch_size=32, seed=0):
    """Run the repo-style workflow: factual filter, pool, split train, then make target-specific sensitive cal/test banks."""
    tokenizer.padding_side = "left"
    pairs_by_family = load_mcqa_pairs(dataset_size=dataset_size, split=split)
    filtered = factual_filter(model, tokenizer, pairs_by_family, device=device, batch_size=batch_size)

    pooled_rows = []
    for family in FAMILIES:
        pooled_rows.extend(filtered[family])

    random.Random(int(seed)).shuffle(pooled_rows)
    if train_pool_size > len(pooled_rows):
        raise ValueError(f"train_pool_size={train_pool_size}, but only {len(pooled_rows)} filtered rows are available")

    train_rows = pooled_rows[:train_pool_size]
    holdout_rows = pooled_rows[train_pool_size:]
    random.Random(f"{int(seed)}:train:shared").shuffle(train_rows)
    train_bank = build_bank(tokenizer, train_rows)

    cal_banks, te_banks = {}, {}
    for target in TARGETS:
        positive_rows = []
        for row in holdout_rows:
            if target == "answer_pointer":
                changed = row["base_answer_pointer"] != row["source_answer_pointer"]
            else:
                changed = row["base_answer_letter"] != row["source_answer_letter"]
            if changed:
                positive_rows.append(row)

        random.Random(f"{int(seed)}:holdout:{target}").shuffle(positive_rows)
        required = cal_size + te_size
        if len(positive_rows) < required:
            raise ValueError(f"target={target} needs {required} holdout rows, but only {len(positive_rows)} changed rows are available")

        cal_banks[target] = build_bank(tokenizer, positive_rows[:cal_size])
        te_banks[target] = build_bank(tokenizer, positive_rows[cal_size:required])
        print(f"[split] target={target} train={len(train_rows)} holdout_candidates={len(holdout_rows)} changed={len(positive_rows)} cal={cal_size} test={te_size}")

    return train_bank, cal_banks, te_banks


if __name__ == "__main__":
    model, tokenizer = load_gemma_model()
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device

    train_bank, cal_banks, te_banks = build_mcqa_banks(
        model=model,
        tokenizer=tokenizer,
        train_pool_size=128,
        cal_size=128,
        te_size=256,
        dataset_size=None,
        split="train",
        device=device,
        batch_size=32,
        seed=0,
    )

    print("train:", train_bank["base_input_ids"].shape)
    for target in TARGETS:
        print(target, "cal:", cal_banks[target]["base_input_ids"].shape, "test:", te_banks[target]["base_input_ids"].shape)
