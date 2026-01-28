import os
import pickle
import json
import torch
from contextlib import nullcontext
from model import GPTConfig, GPT
# On n'utilise plus argparse pour le prompt, mais on peut garder les autres réglages

# --- Configuration ---
out_dir = 'out-5'
input_file = 'questions.txt'  # Ton fichier avec les 50 questions (une par ligne)
output_file = 'responses.json'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
max_new_tokens = 500
temperature = 0.8
top_k = 200

# --- Chargement du Modèle (Identique à ton script original) ---
checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# --- Gestion de l'encodage ---
# On simplifie ici : on vérifie si meta.pkl existe, sinon GPT-2 par défaut
meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# --- Boucle de Génération ---
results = []
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=torch.float16)

if not os.path.exists(input_file):
    print(f"Erreur : Le fichier {input_file} est introuvable.")
else:
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"Début de la génération pour {len(questions)} questions...")

    for i, question in enumerate(questions):
        print(f"Génération {i+1}/{len(questions)} : {question[:30]}...")

        # Encodage de la question
        start_ids = encode(question)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

        with torch.no_grad():
            with ctx:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                full_output = decode(y[0].tolist())

                # On nettoie souvent la sortie pour ne garder que la réponse (optionnel)
                # Si le modèle répète la question, on peut essayer de l'enlever :
                answer = full_output[len(question):].strip()

        results.append({
            "id": str(i + 1),
            "input": question,
            "output": answer
        })

    # --- Sauvegarde en JSON ---
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Terminé ! Les réponses sont dans {output_file}")
