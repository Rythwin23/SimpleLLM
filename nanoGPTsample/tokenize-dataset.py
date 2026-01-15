import os
import pickle

import numpy as np
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
from datasets import load_dataset


def get_text_field(example):
    possible_keys = ['prompt', 'text']  # V2 prise en compte du dataset

    for key in possible_keys:
        if key in example and isinstance(example[key], str):
            return example[key]

    for key, value in example.items():
        if isinstance(value, str) and len(value) > 0:
            return value

    return ""


def tokenize_split(dataset_split, tokenizer, max_samples=None):
    all_tokens = []
    count = 0

    for example in tqdm(dataset_split, desc="Tokenizing"):
        if max_samples and count >= max_samples:
            break

        text = get_text_field(example)
        text = text.strip()
        if not text:  # ajouter en V2
            continue

        if text and len(text) > 0:
            try:
                tokens = tokenizer.encode(text, add_special_tokens=False)  # V1 True, V2 False
                MAX_DOC_TOKENS = 1024  # V1=None, V2=1024, V3=512(la limite de bert)
                tokens = tokens[:MAX_DOC_TOKENS]
                if tokens:
                    all_tokens.extend(tokens)
                    if tokenizer.eos_token_id is not None:  # ajouter en V2
                        all_tokens.append(tokenizer.eos_token_id)  # marche avec gpt2, bert n'a pas de token de fin
                count += 1
            except Exception as e:
                print(f"Error on example {count}: {str(e)[:50]}")
                continue
    dtype = np.uint16 if tokenizer.vocab_size < 65536 else np.uint32  # V1 : uint16, V2 uint32
    tokens_array = np.array(all_tokens, dtype=dtype)
    print(f"Tokenized {count} examples -> {len(tokens_array):,} tokens")

    return tokens_array


def split_train_val(tokens_array, val_split=0.05):
    split_idx = int(len(tokens_array) * (1 - val_split))
    train_tokens = tokens_array[:split_idx]
    val_tokens = tokens_array[split_idx:]

    print(f"Train: {len(train_tokens):,} tokens")
    print(f"Val: {len(val_tokens):,} tokens")

    return train_tokens, val_tokens


def save_tokens(tokens_array, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokens_array.tofile(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved {output_path} ({size_mb:.2f} MB)")


def save_meta(tokenizer, output_dir):
    meta = {
        'vocab_size': tokenizer.vocab_size,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
    }

    meta_path = os.path.join(output_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"Saved meta.pkl (vocab_size: {meta['vocab_size']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HuggingFaceTB/cosmopedia-100k')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default='gpt2')  # V1-V2:gpt2 / V3:bert-base-uncased
    parser.add_argument('--output', type=str, default='data/cosmopedia-100k-v2')
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=None)
    parser.add_argument('--val_split', type=float, default=0.05)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading dataset: {args.dataset}")
    if args.config:
        dataset = load_dataset(args.dataset, args.config)
    else:
        dataset = load_dataset(args.dataset)

    print(f"Available splits: {list(dataset.keys())}")

    print(f"Loading tokenizer: {args.tokenizer}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(args.output, exist_ok=True)

    if 'train' in dataset and 'validation' in dataset:
        print("\nTokenizing train split...")
        train_tokens = tokenize_split(dataset['train'], tokenizer, args.max_train)

        print("\nTokenizing validation split...")
        val_tokens = tokenize_split(dataset['validation'], tokenizer, args.max_val)

    elif 'train' in dataset:
        print("\nTokenizing train split...")
        train_tokens_full = tokenize_split(dataset['train'], tokenizer, args.max_train)

        print("\nSplitting train/val...")
        train_tokens, val_tokens = split_train_val(train_tokens_full, args.val_split)

    else:
        splits = list(dataset.keys())
        print(f"\nTokenizing {splits[0]} split...")
        train_tokens_full = tokenize_split(dataset[splits[0]], tokenizer)

        print("\nSplitting train/val...")
        train_tokens, val_tokens = split_train_val(train_tokens_full, args.val_split)

    print("\nSaving files...")
    save_tokens(train_tokens, os.path.join(args.output, 'train.bin'))
    save_tokens(val_tokens, os.path.join(args.output, 'val.bin'))
    save_meta(tokenizer, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
