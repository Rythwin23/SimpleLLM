import json
import matplotlib.pyplot as plt
import numpy as np

json_file = "out-15/metrics.json"
name = json_file[:-5]

with open(json_file, 'r') as f:
    data = json.load(f)

iterations = [d['iter'] for d in data]
train_loss = [d['train_loss'] for d in data]
val_loss = [d['val_loss'] for d in data]
lr = [d['lr'] for d in data]
tokens_processed = [d['tokens_processed'] / 1e9 for d in data]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Entraînement du modèle - Métriques', fontsize=10, fontweight='bold')

ax1 = axes[0]
ax1.plot(iterations, train_loss, 'o-', label='Training Loss', color='#2E86AB', linewidth=1, markersize=2)
ax1.plot(iterations, val_loss, 's-', label='Validation Loss', color='#A23B72', linewidth=1, markersize=2)
ax1.set_xlabel('Itération', fontsize=6)
ax1.set_ylabel('Loss', fontsize=6)
ax1.set_title('Training vs Validation Loss', fontsize=8, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(iterations, lr, 'o-', color='#F18F01', linewidth=1, markersize=2)
ax2.set_xlabel('Itération', fontsize=6)
ax2.set_ylabel('Learning Rate', fontsize=6)
ax2.set_title('Learning Rate Schedule', fontsize=8, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fname=name, dpi=300, bbox_inches='tight')
plt.show()
