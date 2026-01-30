import pickle
import numpy as np

# Vérifier la capacité du tokenizer utilisé
meta = pickle.load(open('data/cosmopedia-100k-BertCorrect/meta.pkl', 'rb'))
print(f"Vocab size: {meta['vocab_size']}")
print(f"EOS token ID: {meta['eos_token_id']}")

# Vérifier la longueur effective des séquences dans train.bin
data = np.fromfile('data/cosmopedia-100k-BertCorrect/train.bin', dtype=np.uint16)
print(f"Total tokens: {len(data):,}")

# Estimer la longueur moyenne des documents (via comptage des EOS)
eos_id = meta['eos_token_id']
doc_count = np.sum(data == eos_id)
avg_doc_length = len(data) / doc_count if doc_count > 0 else 0
print(f"Documents détectés: {doc_count:,}")
print(f"Longueur moyenne par document: {avg_doc_length:.1f} tokens")