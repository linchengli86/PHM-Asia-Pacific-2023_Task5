"""Visual proof that window cropping captures discriminative information.

Generates 4 figures:
  1. Time series: Normal vs Fault — active vs equilibrium regions
  2. t-SNE: window embeddings colored by task label (active vs equilibrium)
  3. Pairwise distance matrix: how different are windows from same/different classes
  4. Simple classifier: active-only vs equilibrium-only vs mixed — proof that active matters
"""

import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

from src.data.dataset import load_raw_data
from src.data.labels import generate_labels
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)

# ─── Load data ───
X_train, _ = load_raw_data('C:/Users/117/Desktop/dataset', list(range(1, 178)), list(range(178, 179)))
def zscore(X):
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True) + 1e-8
    return (X - m) / s
X = zscore(X_train)
y = generate_labels('C:/Users/117/Desktop/dataset/train/labels.xlsx')

# ─── Extract SSL features (frozen encoder) ───
class CNNE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        c = 7; ls = []
        for ch, k in zip([64, 128, 128], [7, 5, 3]):
            ls.extend([torch.nn.Conv1d(c, ch, k, padding=k//2),
                       torch.nn.BatchNorm1d(ch), torch.nn.ReLU(),
                       torch.nn.MaxPool1d(2), torch.nn.Dropout(0.3)])
            c = ch
        self.net = torch.nn.Sequential(*ls)
        self.gap = torch.nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        return self.gap(self.net(x.transpose(1, 2))).squeeze(-1)

ssl = CNNE().cuda()
ssl.load_state_dict(torch.load('models/final/ssl_encoder.pt', 'cpu'))

# Crop windows from two regions and extract features
def crop_windows(X, region, n_per_case=8, ws=200):
    """Crop windows from a specific region."""
    X_ds = X[:, ::6, :][:, :200, :]
    feats = []
    labels = {"task1": [], "task2": [], "task3": [], "task4": [], "task5": []}
    for i in range(len(X)):
        ts = X_ds[i]
        r_start = max(0, region[0] // 6)
        r_end = min(ts.shape[0], region[1] // 6)
        tp = max(0, r_end - ws)
        for _ in range(n_per_case):
            t0 = np.random.randint(r_start, max(r_start + 1, tp)) if tp > r_start else 0
            t0 = min(t0, max(0, ts.shape[0] - ws))
            w = ts[t0:t0 + ws]
            if w.shape[0] < ws:
                w = np.pad(w, ((0, ws - w.shape[0]), (0, 0)))
            feats.append(w)
            for k in labels:
                labels[k].append(y[k][i])
    X_w = np.stack(feats).astype(np.float32)
    labels = {k: np.array(v) for k, v in labels.items()}
    # Extract SSL features
    embeds = []
    with torch.no_grad():
        for j in range(0, len(X_w), 128):
            xb = torch.from_numpy(X_w[j:j+128]).float().cuda()
            embeds.append(ssl(xb).cpu().numpy())
    return np.concatenate(embeds, axis=0), labels, X_w

# Map task values to display labels
def get_task5_cls(t5_vals):
    """Group task5 into discrete classes for visualization."""
    cls = np.zeros(len(t5_vals), dtype=int)
    cls[t5_vals == 100] = 0   # Normal
    cls[t5_vals == 0] = 1     # 0%
    cls[t5_vals == 25] = 2    # 25%
    cls[t5_vals == 50] = 3    # 50%
    cls[t5_vals == 75] = 4    # 75%
    return cls

print("Extracting active-region features...")
emb_active, lab_active, Xw_active = crop_windows(X, (0, 800), n_per_case=8)
print("Extracting equilibrium-region features...")
emb_eq, lab_eq, Xw_eq = crop_windows(X, (800, 1200), n_per_case=8)

# ─── Figure 1: Time Series — Active vs Equilibrium ───
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
# Normal case, active
n_idx = np.where(lab_active["task1"] == 0)[0][0]
f_idx = np.where(lab_active["task1"] == 1)[0][0]
t = np.arange(200) * 6  # original time scale

for col, (title, region_w, region_lab) in enumerate([
    ("Active Region [0,800]", Xw_active, lab_active),
    ("Equilibrium [800,1200]", Xw_eq, lab_eq),
]):
    n_i = np.where(region_lab["task1"] == 0)[0][0]
    f_i = np.where(region_lab["task1"] == 1)[0][0]
    ax = axes[0, col]
    ax.plot(t, region_w[n_i, :, 0], 'b-', alpha=0.7, lw=1.5, label='Normal (P1)')
    ax.plot(t, region_w[f_i, :, 0], 'r-', alpha=0.7, lw=1.5, label='Fault (P1)')
    ax.set_title(title, fontsize=13, weight='bold')
    ax.set_xlabel('Time step'); ax.set_ylabel('P1 Value')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Distribution of P1 std across all windows
for col, (title, emb, lab) in enumerate([
    ("Active Region", emb_active, lab_active),
    ("Equilibrium", emb_eq, lab_eq),
]):
    ax = axes[1, col]
    norm_std = Xw_active[np.where(lab_active["task1"] == 0)[0], :, 0].std(axis=1)
    fault_std = Xw_active[np.where(lab_active["task1"] == 1)[0], :, 0].std(axis=1)
    if col == 1:
        norm_std = Xw_eq[np.where(lab_eq["task1"] == 0)[0], :, 0].std(axis=1)
        fault_std = Xw_eq[np.where(lab_eq["task1"] == 1)[0], :, 0].std(axis=1)
    ax.hist(norm_std, bins=30, alpha=0.6, color='blue', label=f'Normal (n={len(norm_std)})')
    ax.hist(fault_std, bins=30, alpha=0.6, color='red', label=f'Fault (n={len(fault_std)})')
    ax.set_title(f'P1 Std Distribution — {title}', fontsize=12, weight='bold')
    ax.set_xlabel('P1 Std'); ax.set_ylabel('Count')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle('Figure 1: Active vs Equilibrium — Time Series & Feature Distribution', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'proof_timeseries.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 1 saved.")

# ─── Figure 2: t-SNE colored by task5 ───
print("Running t-SNE...")
# Sample 2000 points for t-SNE speed
n_sample = 2000
idx_a = np.random.choice(len(emb_active), min(n_sample, len(emb_active)), replace=False)
idx_e = np.random.choice(len(emb_eq), min(n_sample, len(emb_eq)), replace=False)
all_emb = np.concatenate([emb_active[idx_a], emb_eq[idx_e]], axis=0)
all_labels = np.concatenate([
    get_task5_cls(lab_active["task5"])[idx_a],
    get_task5_cls(lab_eq["task5"])[idx_e]
])
regions = np.array(["Active"] * len(idx_a) + ["Equilibrium"] * len(idx_e))

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
emb_2d = tsne.fit_transform(all_emb)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#2196F3', '#F44336', '#FF9800', '#4CAF50', '#9C27B0']
names = ['Normal(100)', 'Fault(0)', 'Fault(25)', 'Fault(50)', 'Fault(75)']

for col, (title, mask) in enumerate([
    ("Active Region", regions == "Active"),
    ("Equilibrium", regions == "Equilibrium"),
]):
    ax = axes[col]
    for c in range(5):
        cm = (all_labels == c) & mask
        ax.scatter(emb_2d[cm, 0], emb_2d[cm, 1], c=colors[c], s=10, alpha=0.6, label=names[c])
    ax.set_title(title, fontsize=13, weight='bold')
    ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
    if col == 0: ax.legend(fontsize=8, markerscale=2)

plt.suptitle('Figure 2: t-SNE of Window Embeddings — Active vs Equilibrium (colored by task5 class)', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'proof_tsne.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 2 saved.")

# ─── Figure 3: Pairwise distance matrix ───
from scipy.spatial.distance import pdist, squareform
# Take 100 windows from each of 5 classes
X_dist = []
y_dist = []
for c in range(5):
    mask = get_task5_cls(lab_active["task5"]) == c
    idx = np.where(mask)[0][:100]
    X_dist.append(emb_active[idx])
    y_dist.extend([c] * len(idx))
X_dist = np.concatenate(X_dist, axis=0)
dist = squareform(pdist(X_dist))
# Sort by class label
idx_sort = np.argsort(np.array(y_dist))
dist_sorted = dist[idx_sort][:, idx_sort]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(dist_sorted, cmap='hot', aspect='auto')
ax.set_title('Figure 3: Pairwise Distance Matrix\n(Active region windows, sorted by task5 class)', fontsize=13, weight='bold')
ax.set_xlabel('Window index (sorted by class)'); ax.set_ylabel('Window index (sorted by class)')
# Add class boundary lines
for i in range(1, 5):
    ax.axhline(y=i*100, color='cyan', lw=1.5, linestyle='--')
    ax.axvline(x=i*100, color='cyan', lw=1.5, linestyle='--')
# Annotate classes
for i, name in enumerate(names):
    ax.text(i*100 + 50, -15, name, ha='center', fontsize=8, rotation=45)
plt.colorbar(im, ax=ax, shrink=0.8, label='Euclidean Distance')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'proof_distance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 3 saved.")

# ─── Figure 4: Simple ridge regression — active vs equilibrium ───
print("Running ridge regression comparison...")
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV

t5_cls_a = get_task5_cls(lab_active["task5"])
t5_cls_e = get_task5_cls(lab_eq["task5"])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
regions_data = [
    ("Active Only", emb_active, lab_active["task5"]),
    ("Equilibrium Only", emb_eq, lab_eq["task5"]),
    ("Mixed (Active + Eq)", np.concatenate([emb_active, emb_eq]),
     np.concatenate([lab_active["task5"], lab_eq["task5"]])),
]

results = []
for i, (name, X_feat, y_val) in enumerate(regions_data):
    ax = axes[i]
    model = RidgeCV(alphas=[0.1, 1.0, 10.0])
    scores = cross_val_score(model, X_feat, y_val, cv=5, scoring='r2')
    model.fit(X_feat, y_val)
    y_pred = model.predict(X_feat[:500])
    y_true = y_val[:500]
    ax.scatter(y_true, y_pred, alpha=0.4, s=15)
    ax.plot([0, 100], [0, 100], 'r--', lw=1.5)
    ax.set_title(f'{name}\nR² (5-fold CV) = {scores.mean():.3f} ± {scores.std():.3f}', fontsize=12, weight='bold')
    ax.set_xlabel('True Opening Ratio'); ax.set_ylabel('Predicted Opening Ratio')
    ax.set_xlim(-5, 105); ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    results.append((name, scores.mean(), scores.std()))

plt.suptitle('Figure 4: Ridge Regression Performance — Active vs Equilibrium vs Mixed', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'proof_regression.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 4 saved.")

print(f"\n{'='*60}")
print("PROOF SUMMARY")
print(f"{'='*60}")
for name, mean, std in results:
    print(f"  {name:<25}: R2 = {mean:.4f} +/- {std:.4f}")
print(f"\n  Conclusion: Active-region windows carry {results[0][1]/max(results[1][1],0.001):.0f}x more")
print(f"  discriminative information than equilibrium windows.")
print(f"\nFigures saved to {FIG}/")
