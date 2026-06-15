import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

DATASET_PATH = "jchang153/copycolors_mcqa"
MODEL_NAME = "google/gemma-2-2b"
NUM_CHOICES = 4


import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "cac_cache")   # dataset + model về đây
os.makedirs(CACHE_DIR, exist_ok=True)


import re
 
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
 
 
# ----------------------------- load dataset ----------------------------- #
 
def load_mcqa_dataset(number_of_examples, split="train"):
    """Tai dataset jchang153/copycolors_mcqa va parse tung row thanh dict de hieu.
 
    Moi example tra ve:
        base_prompt            cau hoi goc, ket thuc bang "Answer:"
        base_answer_letter     chu cai dap an dung cua base (vd "D")
        source_prompt_pointer  prompt counterfactual cho answer_pointer (doi thu tu choice)
        source_answer_letter_pointer   dap an dung cua prompt do (vd "B")
        source_prompt_token    prompt counterfactual cho answer_token (doi nhan chu cai)
        source_answer_letter_token     dap an dung cua prompt do (vd "W")
    """
    dataset = load_dataset(DATASET_PATH, split=split, cache_dir=CACHE_DIR)
    dataset = dataset.select(range(number_of_examples))
 
    examples = []
    for row in dataset:
        base_prompt = row["prompt"]
        base_answer_letter = find_answer_letter(base_prompt, row["choices"])
 
        pointer_counterfactual = row["answerPosition_counterfactual"]
        token_counterfactual = row["randomLetter_counterfactual"]
 
        examples.append({
            "base_prompt": base_prompt,
            "base_answer_letter": base_answer_letter,
            "source_prompt_pointer": pointer_counterfactual["prompt"],
            "source_answer_letter_pointer": find_answer_letter(
                pointer_counterfactual["prompt"], pointer_counterfactual["choices"]),
            "source_prompt_token": token_counterfactual["prompt"],
            "source_answer_letter_token": find_answer_letter(
                token_counterfactual["prompt"], token_counterfactual["choices"]),
        })
    return examples
 
 
def find_answer_letter(prompt, choices):
    """Tim chu cai dap an dung: mau duoc hoi trong cau ("An owl is cyan." -> "cyan"),
    roi tra ve label cua choice co text bang mau do."""
    match = re.search(r"\b(?:is|are)\s+(\w+)\.", prompt)
    queried_color = match.group(1)
    position = choices["text"].index(queried_color)
    return choices["label"][position]
 
 
# ----------------------------- load model ------------------------------- #
 
def load_gemma_model(device="cuda"):
    """Tai Gemma-2-2B + tokenizer. Can bien moi truong HF_TOKEN (model gated).
    Tokenizer de left-padding: token that don ve cuoi, nen logits cot cuoi (-1)
    luon la vi tri du doan next-token."""
    huggingface_token = os.environ.get("HF_TOKEN")
 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=huggingface_token, cache_dir=CACHE_DIR)
    tokenizer.padding_side = "left"
 
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=huggingface_token,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=CACHE_DIR
    )
    model.eval()
    return model, tokenizer
 
def letter_token_id(tokenizer, letter):
    """Token id cua mot chu cai dap an. Nhan ca ' A' lan 'A' (lay cai ra dung 1 token)."""
    for candidate in (" " + letter, letter):
        ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"chu cai {letter!r} khong ra 1 token")
 
 
def find_index_of_letter_position_in_prompt(tokenizer, prompt, letter):
    """Index (chua padding) cua token chu cai dap an trong prompt.
    Dap an luon o dang 'D. cyan' -> tim 'D.', roi tokenize phan prompt tinh den
    het chu cai do; token cuoi chinh la chu cai do."""
    char_index = prompt.index(letter + ".")
    prefix_ids = tokenizer(prompt[: char_index + 1], add_special_tokens=True)["input_ids"]
    return len(prefix_ids) - 1
 
 
@torch.no_grad()
def gemma_is_correct(model, tokenizer, prompts, gold_letters, device):
    """Bool list: Gemma doan dung chu cai dap an cho moi prompt khong.
    Left-padding -> token du doan o cot cuoi (-1)."""
    encoding = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    predicted_ids = model(**encoding).logits[:, -1, :].argmax(dim=-1)
    gold_ids = torch.tensor([letter_token_id(tokenizer, l) for l in gold_letters], device=device)
    return (predicted_ids == gold_ids).cpu().tolist()
 
 
def build_bank(model, tokenizer, target_variable, bank_size, device="cuda"):
    """Xay bank cho mot bien: 'answer_pointer' hoac 'answer_token'."""
    source_prompt_key, source_letter_key = {
        "answer_pointer": ("source_prompt_pointer", "source_answer_letter_pointer"),
        "answer_token": ("source_prompt_token", "source_answer_letter_token"),
    }[target_variable]
 
    # 1) lay du lieu (lay du gap 2 de bu phan bi loc)
    examples = load_mcqa_dataset(bank_size * 2)
    base_prompts = [ex["base_prompt"] for ex in examples]
    base_letters = [ex["base_answer_letter"] for ex in examples]
    source_prompts = [ex[source_prompt_key] for ex in examples]
    source_letters = [ex[source_letter_key] for ex in examples]
 
    # 2) loc: giu cap Gemma dung CA base lan source
    base_ok = gemma_is_correct(model, tokenizer, base_prompts, base_letters, device)
    source_ok = gemma_is_correct(model, tokenizer, source_prompts, source_letters, device)
    kept = [i for i in range(len(examples)) if base_ok[i] and source_ok[i]][:bank_size]
    print(f"giu {len(kept)}/{len(examples)} cap (base dung {sum(base_ok)}, source dung {sum(source_ok)})")
 
    def pick(values):
        return [values[i] for i in kept]
 
    # 3) tokenize cac prompt da giu (left-padded)
    base_encoding = tokenizer(pick(base_prompts), padding=True, return_tensors="pt")
    source_encoding = tokenizer(pick(source_prompts), padding=True, return_tensors="pt")
 
    # 4) vi tri token dap an (chua padding) + nhan counterfactual (= dap an source)
    base_positions = [find_index_of_letter_position_in_prompt(tokenizer, base_prompts[i], base_letters[i]) for i in kept]
    source_positions = [find_index_of_letter_position_in_prompt(tokenizer, source_prompts[i], source_letters[i]) for i in kept]
    counterfactual_letters = pick(source_letters)
    counterfactual_ids = [letter_token_id(tokenizer, l) for l in counterfactual_letters]
 
    return {
        "base_input_ids": base_encoding["input_ids"],
        "base_attention_mask": base_encoding["attention_mask"],
        "source_input_ids": source_encoding["input_ids"],
        "source_attention_mask": source_encoding["attention_mask"],
        "base_answer_positions": torch.tensor(base_positions),
        "source_answer_positions": torch.tensor(source_positions),
        "counterfactual_letters": counterfactual_letters,
        "counterfactual_token_ids": torch.tensor(counterfactual_ids),
    }
 
 
if __name__ == "__main__":
 
    model, tokenizer = load_gemma_model()
 
    bank = build_bank(model, tokenizer, "answer_pointer", bank_size=64, device=device)
    print("base_input_ids:", tuple(bank["base_input_ids"].shape))
    print("vi tri dap an base (5 dau):", bank["base_answer_positions"][:5].tolist())
    print("nhan cf (5 dau):", bank["counterfactual_letters"][:5])
