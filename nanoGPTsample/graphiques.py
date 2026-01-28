import json
import matplotlib.pyplot as plt
import numpy as np

json_file = "./out-7/metrics.json"

with open(json_file, 'r') as f:
    data = json.load(f)

iterations = [d['iter'] for d in data]
train_loss = [d['train_loss'] for d in data]
val_loss = [d['val_loss'] for d in data]
lr = [d['lr'] for d in data]
tokens_processed = [d['tokens_processed'] / 1e9 for d in data]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Entraînement du modèle - Métriques', fontsize=16, fontweight='bold')

ax1 = axes[0, 0]
ax1.plot(iterations, train_loss, 'o-', label='Training Loss', color='#2E86AB', linewidth=2, markersize=6)
ax1.plot(iterations, val_loss, 's-', label='Validation Loss', color='#A23B72', linewidth=2, markersize=6)
ax1.set_xlabel('Itération', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(iterations, lr, 'o-', color='#F18F01', linewidth=2, markersize=6)
ax2.set_xlabel('Itération', fontsize=11)
ax2.set_ylabel('Learning Rate', fontsize=11)
ax2.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
ax3.plot(iterations, tokens_processed, 'o-', color='#06A77D', linewidth=2, markersize=6)
ax3.set_xlabel('Itération', fontsize=11)
ax3.set_ylabel('Tokens Traités (Milliards)', fontsize=11)
ax3.set_title('Tokens Traités Cumulatifs', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
gap = [v - t for t, v in zip(train_loss, val_loss)]
ax4.bar(range(len(iterations)), gap, color='#C1121F', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Itération', fontsize=11)
ax4.set_ylabel('Val Loss - Train Loss', fontsize=11)
ax4.set_title('Écart Train/Validation (Overfitting)', fontsize=12, fontweight='bold')
ax4.set_xticks(range(len(iterations)))
ax4.set_xticklabels(iterations, rotation=45)
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig('metrics-Unigram-2.2k.png', dpi=300, bbox_inches='tight')
plt.show()