"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
from transformers import AutoTokenizer  # <-- NOUVEAU
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'out-13'
tokenizer = "BERT"
num_samples = 3
max_new_tokens = 100
temperature = 0.75
top_k = 100
seed = 1337
device = 'cuda:0'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 1. Charger le checkpoint
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)

# 2. Créer le modèle
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

# 3. Charger les poids
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

if compile:
    model = torch.compile(model)

# 4. Charger le BON tokenizer selon le vocab_size
if tokenizer == "BERT":  # BERT
    print("Loading BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    decode = lambda l: tokenizer.decode(l, skip_special_tokens=True)
elif tokenizer == "UNIGRAM":  # BERT
    print("Loading Unigram tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    decode = lambda l: tokenizer.decode(l, skip_special_tokens=True)
else:  # BPE (GPT-2) par défaut
    print("Loading GPT-2 BPE tokenizer...")
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

"""
# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()"""


while True:
    start = str(input("Prompt :"))
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    print(f"\nPrompt: {start}")
    print(f"Encoded tokens: {start_ids}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Vocab size: {model.config.vocab_size}\n")

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print('---------------------------------------------------------')