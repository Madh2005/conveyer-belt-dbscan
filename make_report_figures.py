"""Build IEEE-column-width figures for the report from outputs/ (run run_pipeline.py and the experiments first)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.decomposition import PCA
import config as C, run_pipeline as P

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5, "axes.linewidth": .6})
OUT = C.BASE_DIR / "report_figs"; W = 3.45
df, inc = P.load(); df, X, kd, eps = P.run_dbscan(df, None, C.MIN_SAMPLES)

# Fig 1 - pipeline
fig, ax = plt.subplots(figsize=(W, 2.45)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-0.6, 10.4)
steps = [("Joint sensors: vibration, tension,\ntemperature (10-min samples)", "#dbe9f6"),
         ("Features: raw + 1-h rolling mean/std\nRobustScaler (9-D vector)", "#dbe9f6"),
         ("DBSCAN: eps from k-distance knee,\nMinPts = 20", "#fde9c9"),
         ("Flags: noise (-1) + clusters < 1 % of data", "#fde9c9"),
         ("Alerts: merge flags <= 60 min apart,\nkeep >= 12 points", "#fbd5d5"),
         ("Validation vs. rupture log:\ndetection, lead time, false alarms", "#d9efd9")]
h = 1.35; gap = .42
for i, (t, c) in enumerate(steps):
    y = 10 - (i + 1) * h - i * gap + 0.35
    ax.add_patch(FancyBboxPatch((1.2, y), 7.6, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=c, ec="#444", lw=.6))
    ax.text(5, y + h / 2, t, ha="center", va="center", fontsize=6.6)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((5, y), (5, y - gap), arrowstyle="-|>", mutation_scale=6, lw=.7, color="#333"))
plt.savefig(OUT / "fig1_pipeline.png", dpi=300, bbox_inches="tight", pad_inches=.03); plt.close()

# Fig 2 - k-distance
fig, ax = plt.subplots(figsize=(W, 2.0)); ax.plot(kd, lw=.9); ax.axhline(eps, color="r", ls="--", lw=.8, label=f"eps = {eps:.2f}")
ax.set_xlabel("Readings (sorted by k-distance)"); ax.set_ylabel(f"Distance to {C.MIN_SAMPLES}th neighbour"); ax.legend(frameon=False)
plt.tight_layout(pad=.3); plt.savefig(OUT / "fig2_kdist.png", dpi=300); plt.close()

# Fig 3 - PCA clusters
p = PCA(2).fit(X); Z = p.transform(X); ev = p.explained_variance_ratio_
fig, ax = plt.subplots(figsize=(W, 2.5)); m = (df.cluster == -1).values
ax.scatter(Z[~m, 0], Z[~m, 1], s=1.5, c="#4a7ab8", alpha=.5, label=f"Cluster 0 (normal), n={int((~m).sum())}", rasterized=True)
ax.scatter(Z[m, 0], Z[m, 1], s=4, c="#d62728", alpha=.8, label=f"Noise / anomalous, n={int(m.sum())}", rasterized=True)
ax.set_xlabel(f"PC1 ({ev[0]*100:.0f} % variance)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.0f} % variance)"); ax.legend(frameon=False, markerscale=3)
plt.tight_layout(pad=.3); plt.savefig(OUT / "fig3_pca.png", dpi=300); plt.close()

# Fig 4 - timeline of one ruptured joint
r = inc.sort_values("rupture_time").iloc[0]; g = df[df.joint_id == r.joint_id]; f = g[g.flag]
fig, ax = plt.subplots(3, 1, figsize=(W, 3.5), sharex=True)
for a, col, lab in zip(ax, C.SENSOR_COLS, ["Vibration (mm/s)", "Tension (kN)", "Temp. (deg C)"]):
    a.plot(g.timestamp, g[col], lw=.4, c="#4a7ab8"); a.scatter(f.timestamp, f[col], s=1.5, c="#d62728", zorder=3)
    a.axvline(r.rupture_time, c="k", ls="--", lw=.8); a.set_ylabel(lab, fontsize=6.5); a.tick_params(labelsize=6)
ax[-1].set_xlabel("Time"); fig.autofmt_xdate(rotation=0, ha="center")
for l in ax[-1].get_xticklabels(): l.set_fontsize(6)
ax[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b")); plt.tight_layout(pad=.3)
plt.savefig(OUT / "fig4_timeline.png", dpi=300); plt.close()

# Fig 5 - method comparison strong vs weak
s = pd.read_csv(C.OUT_DIR / "table_methods.csv"); w = pd.read_csv(C.OUT_DIR / "table_weak.csv")
names = ["DBSCAN", "IForest", "LOF", "k-means", "Threshold"]; x = np.arange(5)
fig, ax = plt.subplots(figsize=(W, 2.1)); ax.bar(x - .2, s.det_rate * 100, .4, label="Strong precursor", color="#4a7ab8")
ax.bar(x + .2, w.det_rate * 100, .4, label="Weak precursor (x0.3)", color="#e8963c")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.5); ax.set_ylabel("Detection rate (%)"); ax.set_ylim(0, 118); ax.legend(frameon=False, ncol=2, loc="upper center")
plt.tight_layout(pad=.3); plt.savefig(OUT / "fig5_methods.png", dpi=300); plt.close()

# Fig 6 - eps sensitivity
t = pd.read_csv(C.OUT_DIR / "table_sensitivity.csv"); e = t[t.param == "eps x knee"]
fig, ax = plt.subplots(figsize=(W, 2.0)); ax.errorbar(e.value, e.lead_mean, yerr=e.lead_std, marker="o", ms=3, lw=.9, c="#4a7ab8", capsize=2)
ax.set_xlabel("eps as multiple of k-distance knee"); ax.set_ylabel("Mean lead time (h)", color="#4a7ab8")
ax2 = ax.twinx(); ax2.plot(e.value, e.det_rate * 100, marker="s", ms=3, lw=.9, c="#d62728", ls="--"); ax2.set_ylabel("Detection rate (%)", color="#d62728"); ax2.set_ylim(0, 110)
plt.tight_layout(pad=.3); plt.savefig(OUT / "fig6_eps.png", dpi=300); plt.close()
print("figures saved to", OUT)
