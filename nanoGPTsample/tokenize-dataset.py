import os
import pickle
import numpy as np
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
from datasets import load_dataset


def get_text_field(example):
    """Extrait le champ texte pertinent du dataset Cosmopedia"""
    # Cosmopedia utilise principalement le champ 'text'
    if 'text' in example and isinstance(example['text'], str) and example['text'].strip():
        return example['text'].strip()

    # Fallback sur d'autres champs possibles
    for key in ['prompt', 'content', 'article']:
        if key in example and isinstance(example[key], str) and example[key].strip():
            return example[key].strip()

    # Dernier recours : chercher n'importe quel champ string non vide
    for key, value in example.items():
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def tokenize_split(dataset_split, tokenizer, max_samples=None):
    all_tokens = []
    count = 0

    # VÉRIFICATION CRITIQUE : EOS token doit exister
    if tokenizer.eos_token_id is None:
        raise ValueError(
            f"Tokenizer '{tokenizer.name_or_path}' n'a pas de EOS token défini !\n"
            f"  → Pour BERT: utiliser [SEP] (ID={tokenizer.sep_token_id})\n"
            f"  → Pour XLNet: utiliser [EOS] (ID={tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 'N/A'})\n"
            f"  → Recommandation: utiliser 'gpt2' ou configurer manuellement le tokenizer"
        )

    for example in tqdm(dataset_split, desc="Tokenizing"):
        if max_samples and count >= max_samples:
            break

        text = get_text_field(example)
        if not text:
            continue

        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)

            # Troncature adaptée à la capacité du tokenizer
            MAX_DOC_TOKENS = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 512
            tokens = tokens[:MAX_DOC_TOKENS]

            if tokens:
                all_tokens.extend(tokens)
                # Ajout sécurisé de l'EOS token (vérifié plus haut)
                all_tokens.append(tokenizer.eos_token_id)
                count += 1

        except Exception as e:
            print(f"Erreur sur exemple {count}: {str(e)[:100]}")
            continue

    # DÉTECTION DE CORRUPTION : vérifier qu'aucun None n'est présent
    if None in all_tokens:
        raise RuntimeError(
            "CORRUPTION DÉTECTÉE: tokens contiennent des valeurs None !\n"
            "→ Cause probable: tokenizer.eos_token_id = None\n"
            "→ Solution: configurer manuellement EOS token avant tokenization"
        )

    dtype = np.uint16 if tokenizer.vocab_size < 65536 else np.uint32
    tokens_array = np.array(all_tokens, dtype=dtype)

    print(f"Tokenized {count:,} examples → {len(tokens_array):,} tokens (dtype={dtype.__name__})")
    print(f"Tokens uniques: {len(np.unique(tokens_array))}/{tokenizer.vocab_size}")
    print(f"EOS token ID utilisé: {tokenizer.eos_token_id} ('{tokenizer.eos_token}')")

    return tokens_array


def split_train_val(tokens_array, val_split):
    split_idx = int(len(tokens_array) * (1 - val_split))
    train_tokens = tokens_array[:split_idx]
    val_tokens = tokens_array[split_idx:]

    print(f"Train: {len(train_tokens):,} tokens ({100 * (1 - val_split):.1f}%)")
    print(f"Val:   {len(val_tokens):,} tokens ({100 * val_split:.1f}%)")

    return train_tokens, val_tokens


def save_tokens(tokens_array, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokens_array.tofile(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Saved {output_path} ({size_mb:.2f} MB)")


def save_meta(tokenizer, output_dir):
    meta = {
        'vocab_size': tokenizer.vocab_size,
        'eos_token_id': tokenizer.eos_token_id,
        'eos_token': tokenizer.eos_token,
        'pad_token_id': tokenizer.pad_token_id,
        'pad_token': tokenizer.pad_token,
        'tokenizer_name': tokenizer.name_or_path,
        'model_max_length': getattr(tokenizer, 'model_max_length', None),
    }
    meta_path = os.path.join(output_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"✓ Saved meta.pkl avec vocab_size={meta['vocab_size']}")


def configure_tokenizer(tokenizer_name):
    """Configure correctement le tokenizer avec EOS/PAD tokens"""
    print(f" Chargement du tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # CORRECTION CRITIQUE POUR BERT
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is not None:
            print(f" EOS token non défini → utilisation de [SEP] (ID={tokenizer.sep_token_id})")
            tokenizer.eos_token = tokenizer.sep_token
            tokenizer.eos_token_id = tokenizer.sep_token_id
        elif tokenizer.cls_token is not None:
            print(f" EOS token non défini → utilisation de [CLS] (ID={tokenizer.cls_token_id})")
            tokenizer.eos_token = tokenizer.cls_token
            tokenizer.eos_token_id = tokenizer.cls_token_id
        else:
            raise ValueError(f"Impossible de définir EOS token pour {tokenizer_name}")

    # Configuration du PAD token
    if tokenizer.pad_token is None:
        print(f" PAD token non défini → utilisation de EOS (ID={tokenizer.eos_token_id})")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"EOS token: '{tokenizer.eos_token}' (ID={tokenizer.eos_token_id})")
    print(f"PAD token: '{tokenizer.pad_token}' (ID={tokenizer.pad_token_id})")
    print(f"Vocab size: {tokenizer.vocab_size}")

    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Tokenize Cosmopedia dataset for nanoGPT")
    parser.add_argument('--dataset', type=str, default='HuggingFaceTB/cosmopedia-100k')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased',
                        help="Tokenizer à utiliser (ex: 'gpt2', 'bert-base-uncased', 'xlnet-base-cased')")
    parser.add_argument('--output', type=str, default='data/cosmopedia-100k-BertCorrect')
    parser.add_argument('--max_train', type=int, default=None, help="Max examples pour train")
    parser.add_argument('--max_val', type=int, default=None, help="Max examples pour val")
    parser.add_argument('--val_split', type=float, default=0.15, help="Fraction pour validation")
    args = parser.parse_args()

    # Configuration sécurisée du tokenizer
    tokenizer = configure_tokenizer(args.tokenizer)

    # Chargement du dataset
    print(f"\n Chargement du dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, args.config) if args.config else load_dataset(args.dataset)
    print(f"✓ Splits disponibles: {list(dataset.keys())}")

    # Inspection rapide de la structure
    sample = next(iter(dataset['train']))
    print(f"✓ Champs disponibles: {list(sample.keys())}")
    print(f"✓ Exemple de texte (premiers 100 chars): '{get_text_field(sample)[:100]}...'")

    os.makedirs(args.output, exist_ok=True)

    # Tokenization selon la structure du dataset
    if 'train' in dataset and 'validation' in dataset:
        print("\n Tokenizing train split...")
        train_tokens = tokenize_split(dataset['train'], tokenizer, args.max_train)

        print("\n Tokenizing validation split...")
        val_tokens = tokenize_split(dataset['validation'], tokenizer, args.max_val)

    elif 'train' in dataset:
        print("\n Tokenizing full train split...")
        train_tokens_full = tokenize_split(dataset['train'], tokenizer, args.max_train)

        print("\n Splitting train/val...")
        train_tokens, val_tokens = split_train_val(train_tokens_full, args.val_split)

    else:
        splits = list(dataset.keys())
        print(f"\n Tokenizing {splits[0]} split...")
        train_tokens_full = tokenize_split(dataset[splits[0]], tokenizer)

        print("\n Splitting train/val...")
        train_tokens, val_tokens = split_train_val(train_tokens_full, args.val_split)

    # Sauvegarde
    print("\n Sauvegarde des fichiers...")
    save_tokens(train_tokens, os.path.join(args.output, 'train.bin'))
    save_tokens(val_tokens, os.path.join(args.output, 'val.bin'))
    save_meta(tokenizer, args.output)

    # Vérification finale de corruption
    print("\n Vérification de corruption des données...")
    train_data = np.fromfile(os.path.join(args.output, 'train.bin'),
                             dtype=np.uint16 if tokenizer.vocab_size < 65536 else np.uint32)
    zeros_pct = np.sum(train_data == 0) / len(train_data) * 100
    eos_count = np.sum(train_data == tokenizer.eos_token_id)

    print(f"Tokens 0 détectés: {zeros_pct:.2f}% (normal si < 5%, suspect si > 10%)")
    print(f"Tokens EOS détectés: {eos_count:,} ({eos_count / len(train_data) * 100:.2f}%)")

    if zeros_pct > 10.0:
        print("ATTENTION: Taux élevé de tokens 0 → possible corruption des données !")
        print("→ Vérifiez que tokenizer.eos_token_id n'était pas None pendant la tokenization")
    else:
        print("Dataset semble correct (taux de tokens 0 acceptable)")

    print("\nTokenization terminée avec succès !")


if __name__ == "__main__":
    main()