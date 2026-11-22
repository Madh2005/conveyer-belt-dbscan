"""DBSCAN anomaly clustering of conveyor-belt joint sensors + validation against rupture logs.
Usage:  python run_pipeline.py [--generate] [--eps 1.2] [--min-samples 20] [--lookback-hours 48]"""
import argparse, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import config as C


# ---------- 1. Load + features ----------
def load():
    s = pd.read_csv(C.SENSOR_CSV, parse_dates=["timestamp"]).sort_values(["joint_id", "timestamp"]).reset_index(drop=True)
    inc = pd.read_csv(C.INCIDENT_CSV, parse_dates=["rupture_time"])
    return s, inc


def build_features(df):
    """Raw readings + rolling mean + rolling std per joint (captures drift and instability)."""
    g = df.groupby("joint_id")[C.SENSOR_COLS]
    mean = g.transform(lambda x: x.rolling(C.ROLL_WINDOW, min_periods=1).mean()).add_suffix("_rmean")
    std = g.transform(lambda x: x.rolling(C.ROLL_WINDOW, min_periods=2).std()).fillna(0).add_suffix("_rstd")
    return pd.concat([df[C.SENSOR_COLS], mean, std], axis=1)


# ---------- 2. DBSCAN ----------
def auto_eps(X, k):
    """Knee of the sorted k-distance curve (point farthest from the chord)."""
    d = np.sort(NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)[0][:, -1])
    x = np.linspace(0, 1, len(d)); y = (d - d[0]) / (d[-1] - d[0] + 1e-12)
    knee = np.argmax(x - y)                      # distance below the diagonal chord
    return float(d[knee]), d


def run_dbscan(df, eps, min_samples):
    feats = build_features(df)
    X = RobustScaler().fit_transform(feats)
    knee_eps, kd = auto_eps(X, min_samples)
    eps_used = eps if eps else knee_eps * C.EPS_SCALE
    labels = DBSCAN(eps=eps_used, min_samples=min_samples, n_jobs=-1).fit_predict(X)
    df = df.copy(); df["cluster"] = labels

    sizes = df.cluster.value_counts()
    small = [c for c, n in sizes.items() if c != -1 and n < C.MIN_CLUSTER_FRAC * len(df)]
    df["flag"] = df.cluster.eq(-1) | df.cluster.isin(small)
    return df, X, kd, eps_used


# ---------- 3. Alerts + validation ----------
def make_alerts(df):
    """Merge nearby flagged points per joint into alerts; ignore short blips (sensor glitches)."""
    alerts = []
    for joint, g in df[df.flag].groupby("joint_id"):
        t = g.timestamp.sort_values().reset_index(drop=True)
        grp = (t.diff() > pd.Timedelta(minutes=C.ALERT_GAP_MIN)).cumsum()
        for _, tt in t.groupby(grp):
            if len(tt) >= C.MIN_ALERT_POINTS:
                alerts.append({"joint_id": joint, "start": tt.iloc[0], "end": tt.iloc[-1], "n_points": len(tt)})
    return pd.DataFrame(alerts, columns=["joint_id", "start", "end", "n_points"])


def validate(df, alerts, inc, lookback_h):
    lb = pd.Timedelta(hours=lookback_h)
    # point-level ground truth: inside pre-rupture window of the joint's incident
    df["true_pre_rupture"] = False
    for _, r in inc.iterrows():
        m = (df.joint_id == r.joint_id) & (df.timestamp >= r.rupture_time - lb) & (df.timestamp <= r.rupture_time)
        df.loc[m, "true_pre_rupture"] = True
    tp = (df.flag & df.true_pre_rupture).sum(); fp = (df.flag & ~df.true_pre_rupture).sum()
    fn = (~df.flag & df.true_pre_rupture).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)

    rows = []
    for _, r in inc.iterrows():
        a = alerts[(alerts.joint_id == r.joint_id) & (alerts.end >= r.rupture_time - lb) & (alerts.start <= r.rupture_time)]
        if len(a):
            first = a.start.min(); lead = (r.rupture_time - max(first, r.rupture_time - lb)).total_seconds() / 3600
            rows.append({**r.to_dict(), "detected": True, "first_alert": first, "lead_time_h": round(lead, 1)})
        else:
            rows.append({**r.to_dict(), "detected": False, "first_alert": None, "lead_time_h": None})
    res = pd.DataFrame(rows)

    # alert-level: alerts that don't overlap any pre-rupture window = false alarms
    def is_true(a):
        m = inc[inc.joint_id == a.joint_id]
        return any((a.end >= r - lb) and (a.start <= r) for r in m.rupture_time)
    alerts = alerts.copy(); alerts["true_alert"] = alerts.apply(is_true, axis=1) if len(alerts) else []
    joint_days = df.joint_id.nunique() * (df.timestamp.max() - df.timestamp.min()).days

    summary = {
        "incidents": int(len(inc)), "incidents_detected": int(res.detected.sum()),
        "detection_rate": round(res.detected.mean(), 3),
        "mean_lead_time_h": round(float(res.lead_time_h.dropna().mean()), 1) if res.detected.any() else None,
        "alerts_total": int(len(alerts)), "false_alarms": int((~alerts.true_alert).sum()) if len(alerts) else 0,
        "false_alarms_per_joint_day": round(((~alerts.true_alert).sum() if len(alerts) else 0) / max(joint_days, 1), 4),
        "point_precision": round(prec, 3), "point_recall": round(rec, 3),
        "point_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3),
    }
    return res, alerts, summary


# ---------- 4. Plots ----------
def plots(df, X, kd, eps, inc):
    C.OUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 4)); plt.plot(kd); plt.axhline(eps, color="r", ls="--", label=f"eps={eps:.2f}")
    plt.title("k-distance curve"); plt.xlabel("points (sorted)"); plt.ylabel("k-distance"); plt.legend()
    plt.tight_layout(); plt.savefig(C.OUT_DIR / "k_distance.png", dpi=130); plt.close()

    p = PCA(2).fit_transform(X); plt.figure(figsize=(7, 5))
    for c in sorted(df.cluster.unique()):
        m = (df.cluster == c).values
        plt.scatter(p[m, 0], p[m, 1], s=4 if c != -1 else 10, c="red" if c == -1 else None,
                    label="noise (anomaly)" if c == -1 else f"cluster {c}", alpha=.6)
    plt.legend(markerscale=3); plt.title("DBSCAN clusters (PCA projection)")
    plt.tight_layout(); plt.savefig(C.OUT_DIR / "clusters_pca.png", dpi=130); plt.close()

    for _, r in inc.iterrows():
        g = df[df.joint_id == r.joint_id]
        fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        for a, col in zip(ax, C.SENSOR_COLS):
            a.plot(g.timestamp, g[col], lw=.6); f = g[g.flag]
            a.scatter(f.timestamp, f[col], c="red", s=6); a.set_ylabel(col)
            a.axvline(r.rupture_time, color="k", ls="--")
        ax[0].set_title(f"{r.joint_id} - {r.incident_id} (red = DBSCAN flagged, dashed = rupture)")
        plt.tight_layout(); plt.savefig(C.OUT_DIR / f"timeline_{r.joint_id}.png", dpi=110); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="create synthetic data first")
    ap.add_argument("--eps", type=float, default=C.EPS)
    ap.add_argument("--min-samples", type=int, default=C.MIN_SAMPLES)
    ap.add_argument("--lookback-hours", type=int, default=C.LOOKBACK_HOURS)
    a = ap.parse_args()
    if a.generate or not C.SENSOR_CSV.exists():
        import simulate_data; simulate_data.simulate()

    df, inc = load()
    df, X, kd, eps = run_dbscan(df, a.eps, a.min_samples)
    alerts = make_alerts(df)
    res, alerts, summary = validate(df, alerts, inc, a.lookback_hours)
    summary.update(eps=round(eps, 3), min_samples=a.min_samples, n_clusters=int(df.cluster.nunique() - (-1 in df.cluster.values)),
                   flagged_points=int(df.flag.sum()), total_points=int(len(df)))

    C.OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(C.OUT_DIR / "labeled_readings.csv", index=False)
    alerts.to_csv(C.OUT_DIR / "alerts.csv", index=False)
    res.to_csv(C.OUT_DIR / "incident_validation.csv", index=False)
    json.dump(summary, open(C.OUT_DIR / "summary.json", "w"), indent=2, default=str)
    plots(df, X, kd, eps, inc)

    print("\n=== Incident validation ===\n", res.to_string(index=False))
    print("\n=== Summary ===");  [print(f"{k:28s}{v}") for k, v in summary.items()]
    print(f"\nOutputs saved in {C.OUT_DIR}")


if __name__ == "__main__":
    main()
