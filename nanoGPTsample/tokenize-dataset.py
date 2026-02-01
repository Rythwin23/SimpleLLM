import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from datasets import load_dataset


def load_tokenizer(name):
    # Charge un tokenizer et retourne un dict avec encode, eos_id, vocab_size
    if name == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return {
            "encode": lambda s: enc.encode(s, allowed_special={"<|endoftext|>"}),
            "eos_id": enc.eot_token,
            "vocab_size": enc.n_vocab,
            "name": "gpt2"
        }
    else:
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(name)
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer '{name}': {e}")

        # Ensure pad token exists
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.sep_token or tok.unk_token

        # Determine EOS token ID with fallback
        eos_id = tok.eos_token_id
        if eos_id is None:
            eos_id = tok.sep_token_id
        if eos_id is None:
            # Last resort: use 0 or vocab_size - 1
            print(f"Warning: No EOS token found for {name}, using token ID 0")
            eos_id = 0

        return {
            "encode": lambda s: tok.encode(s, add_special_tokens=False),
            "eos_id": eos_id,
            "vocab_size": len(tok),  # Plus fiable que tok.vocab_size
            "name": name
        }


def get_text(example):
    # Extrait le texte d'un exemple de dataset
    # les clés de cosmopedia
    for k in ["text"]:
        if k in example and isinstance(example[k], str) and example[k].strip():
            return example[k].strip()
    return ""


def tokenize_split(dataset, tokenizer, max_docs):
    # Tokenize un dataset et retourne un array numpy
    tokens = []
    count = 0
    empty_docs = 0

    for ex in tqdm(dataset, desc="Tokenizing"):
        if max_docs and count >= max_docs:
            break

        text = get_text(ex)
        if not text:
            empty_docs += 1
            continue

        try:
            ids = tokenizer["encode"](text)
        except Exception as e:
            print(f"Skipping doc {count}: tokenization error: {e}")
            continue

        if ids:
            tokens.extend(ids)
            tokens.append(tokenizer["eos_id"])
            count += 1

    if empty_docs > 0:
        print(f"Skipped {empty_docs} empty documents")

    if not tokens:
        raise ValueError("No tokens generated! Check your dataset.")

    # Choisir le bon dtype
    dtype = np.uint16 if tokenizer["vocab_size"] < 65536 else np.uint32
    return np.array(tokens, dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description="Tokenize datasets for LLM training")
    parser.add_argument("--dataset", default="HuggingFaceTB/cosmopedia-100k",
                        help="HuggingFace dataset name")
    parser.add_argument("--tokenizer", default="albert-base-v2",
                        help="Tokenizer: gpt2 | bert-base-uncased | albert-base-v2 | etc.")
    parser.add_argument("--out", default="data/cosmopedia-100k-Unigram",
                        help="Output directory")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.05 = 5%)")
    parser.add_argument("--max_docs", type=int, default=None,
                        help="Max number of documents to process (for testing)")

    args = parser.parse_args()

    print("=" * 60)
    print("TOKENIZATION CONFIGURATION")
    print(f"Dataset: {args.dataset}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Output: {args.out}")
    print(f"Val split: {args.val_split * 100}%")
    print(f"Max docs: {args.max_docs or 'All'}")
    print("=" * 60)

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"\nTokenizer loaded: {tokenizer['name']}")
    print(f"  - Vocab size: {tokenizer['vocab_size']:,}")
    print(f"  - EOS token ID: {tokenizer['eos_id']}")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}...")
    try:
        dataset = load_dataset(args.dataset, split="train")
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")

    print(f"✓ Dataset loaded: {len(dataset):,} examples")

    # Tokenize
    print(f"\nTokenizing...")
    all_tokens = tokenize_split(dataset, tokenizer, args.max_docs)

    print(f"\nTokenization complete!")
    print(f"  - Total tokens: {len(all_tokens):,}")
    print(f"  - Size: {len(all_tokens) * all_tokens.itemsize / 1024 ** 2:.2f} MB")

    # Split train/val
    split_idx = int(len(all_tokens) * (1 - args.val_split))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    print(f"\nSplit:")
    print(f"  - Train: {len(train_tokens):,} tokens ({len(train_tokens) * train_tokens.itemsize / 1024 ** 2:.2f} MB)")
    print(f"  - Val:   {len(val_tokens):,} tokens ({len(val_tokens) * val_tokens.itemsize / 1024 ** 2:.2f} MB)")

    # Save files
    os.makedirs(args.out, exist_ok=True)

    train_path = os.path.join(args.out, "train.bin")
    val_path = os.path.join(args.out, "val.bin")
    meta_path = os.path.join(args.out, "meta.pkl")

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    meta = {
        "vocab_size": tokenizer["vocab_size"],
        "eos_token_id": tokenizer["eos_id"],
        "tokenizer": tokenizer["name"]
    }

    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"\nFiles saved:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {meta_path}")

    print("\n" + "=" * 60)
    print("TOKENIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()