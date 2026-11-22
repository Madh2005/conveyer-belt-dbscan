"""Stress test: weak precursor (amplitude x0.3). Usage: python experiments_weak.py 0 10"""
import sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN
import config as C, simulate_data, run_pipeline as P
tmp = Path(tempfile.mkdtemp()); C.DATA_DIR = tmp; C.SENSOR_CSV = tmp/"s.csv"; C.INCIDENT_CSV = tmp/"i.csv"
C.PRECURSOR_SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
A, B = int(sys.argv[1]), int(sys.argv[2]); rows = []
def score(df, flag, inc):
    df = df.copy(); df["flag"] = flag; al = P.make_alerts(df)
    _, al, s = P.validate(df, al, inc, C.LOOKBACK_HOURS)
    return {"det": s["detection_rate"], "lead": s["mean_lead_time_h"], "fa": s["false_alarms"], "alerts": s["alerts_total"],
            "n_inc": s["incidents"], "n_det": s["incidents_detected"]}
for sd in range(A, B):
    C.SEED = sd; simulate_data.simulate(); df, inc = P.load()
    X = RobustScaler().fit_transform(P.build_features(df))
    knee, _ = P.auto_eps(X, C.MIN_SAMPLES); lab = DBSCAN(eps=knee, min_samples=C.MIN_SAMPLES, n_jobs=-1).fit_predict(X)
    sizes = pd.Series(lab).value_counts(); small = [c for c, n in sizes.items() if c != -1 and n < C.MIN_CLUSTER_FRAC*len(lab)]
    fl = (lab == -1) | np.isin(lab, small); frac = fl.mean()
    rows.append({"seed": sd, "method": "DBSCAN (proposed)", **score(df, fl, inc)})
    rows.append({"seed": sd, "method": "Isolation Forest", **score(df, IsolationForest(n_estimators=200, contamination=frac, random_state=sd).fit_predict(X) == -1, inc)})
    rows.append({"seed": sd, "method": "LOF", **score(df, LocalOutlierFactor(n_neighbors=20, contamination=frac).fit_predict(X) == -1, inc)})
    d = KMeans(3, n_init=10, random_state=sd).fit(X).transform(X).min(1)
    rows.append({"seed": sd, "method": "k-means distance", **score(df, d > np.quantile(d, 1-frac), inc)})
    med = np.median(df[C.SENSOR_COLS], 0); mad = np.median(np.abs(df[C.SENSOR_COLS]-med), 0)*1.4826
    rows.append({"seed": sd, "method": "Fixed threshold (|z|>3.5)", **score(df, (np.abs((df[C.SENSOR_COLS]-med)/mad) > 3.5).any(axis=1).values, inc)})
    print("seed", sd, flush=True)
pd.DataFrame(rows).to_csv(C.OUT_DIR/f"exp_weak_{A}.csv", index=False)
