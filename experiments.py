"""Multi-seed benchmark: DBSCAN vs baselines + sensitivity + feature ablation.
Usage: python experiments.py   (about 1-3 minutes). Results -> outputs/exp_*.csv"""
import time, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
import config as C
import simulate_data, run_pipeline as P

import sys
A, B = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 10)
SEEDS = list(range(A, B))
tmp = Path(tempfile.mkdtemp()); C.DATA_DIR = tmp
C.SENSOR_CSV = tmp / "s.csv"; C.INCIDENT_CSV = tmp / "i.csv"


def score(df, flag, inc):
    df = df.copy(); df["flag"] = flag
    alerts = P.make_alerts(df)
    res, alerts, s = P.validate(df, alerts, inc, C.LOOKBACK_HOURS)
    return {"det": s["detection_rate"], "lead": s["mean_lead_time_h"], "fa": s["false_alarms"],
            "alerts": s["alerts_total"], "n_inc": s["incidents"], "n_det": s["incidents_detected"], "f1pt": s["point_f1"]}


_cache = {}
def knee_dbscan(X, ms, scale=1.0):
    key = (id(X), ms)
    if key not in _cache: _cache[key] = P.auto_eps(X, ms)
    knee, _ = _cache[key]
    lab = DBSCAN(eps=knee * scale, min_samples=ms, n_jobs=-1).fit_predict(X)
    sizes = pd.Series(lab).value_counts()
    small = [c for c, n in sizes.items() if c != -1 and n < C.MIN_CLUSTER_FRAC * len(lab)]
    return (lab == -1) | np.isin(lab, small)


rows, sens, abl = [], [], []
for sd in SEEDS:
    C.SEED = sd; simulate_data.simulate()
    df, inc = P.load()
    Xf = RobustScaler().fit_transform(P.build_features(df))
    Xr = RobustScaler().fit_transform(df[C.SENSOR_COLS])

    t0 = time.time(); flag = knee_dbscan(Xf, C.MIN_SAMPLES); rt = time.time() - t0
    frac = flag.mean()
    rows.append({"seed": sd, "method": "DBSCAN (proposed)", "frac": frac, "runtime_s": rt, **score(df, flag, inc)})

    iso = IsolationForest(n_estimators=200, contamination=frac, random_state=sd).fit_predict(Xf) == -1
    rows.append({"seed": sd, "method": "Isolation Forest", "frac": frac, **score(df, iso, inc)})
    lof = LocalOutlierFactor(n_neighbors=20, contamination=frac).fit_predict(Xf) == -1
    rows.append({"seed": sd, "method": "LOF", "frac": frac, **score(df, lof, inc)})
    km = KMeans(3, n_init=10, random_state=sd).fit(Xf); d = km.transform(Xf).min(1)
    rows.append({"seed": sd, "method": "k-means distance", "frac": frac, **score(df, d > np.quantile(d, 1 - frac), inc)})
    med = np.median(df[C.SENSOR_COLS], 0); mad = np.median(np.abs(df[C.SENSOR_COLS] - med), 0) * 1.4826
    thr = (np.abs((df[C.SENSOR_COLS] - med) / mad) > 3.5).any(axis=1).values
    rows.append({"seed": sd, "method": "Fixed threshold (|z|>3.5)", "frac": thr.mean(), **score(df, thr, inc)})

    # ablation: raw features only
    fr = knee_dbscan(Xr, 6)
    abl.append({"seed": sd, "variant": "Raw readings only", **score(df, fr, inc)})
    abl.append({"seed": sd, "variant": "Raw + rolling (proposed)", **score(df, flag, inc)})
    # sensitivity
    for m in [0.8, 1.0, 1.25, 1.5, 2.0]:
        sens.append({"seed": sd, "param": "eps x knee", "value": m, **score(df, knee_dbscan(Xf, C.MIN_SAMPLES, m), inc)})
    for ms in [10, 20, 40]:
        sens.append({"seed": sd, "param": "min_samples", "value": ms, **score(df, knee_dbscan(Xf, ms), inc)})
    for mp in [6, 12, 18]:
        C.MIN_ALERT_POINTS = mp
        sens.append({"seed": sd, "param": "min_alert_points", "value": mp, **score(df, flag, inc)})
    C.MIN_ALERT_POINTS = 12
    _cache.clear()
    print("seed", sd, "done", flush=True)

C.OUT_DIR.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(C.OUT_DIR / f"exp_methods_{A}.csv", index=False)
pd.DataFrame(sens).to_csv(C.OUT_DIR / f"exp_sensitivity_{A}.csv", index=False)
pd.DataFrame(abl).to_csv(C.OUT_DIR / f"exp_ablation_{A}.csv", index=False)
print("saved chunk", A, B)
