"""Merge experiment chunks and print summary tables."""
import glob, pandas as pd, config as C
def load(n): return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(C.OUT_DIR / f"exp_{n}_*.csv")))])
def agg(d, key):
    out = d.groupby(key, sort=False).agg(n_inc=("n_inc","sum"), n_det=("n_det","sum"), lead_mean=("lead","mean"),
        lead_std=("lead","std"), fa_mean=("fa","mean"), fa_std=("fa","std"), alerts=("alerts","mean"), runs=("seed","nunique"))
    out["det_rate"] = out.n_det / out.n_inc
    return out.round(3)
if __name__ == "__main__":
    m, a, s = load("methods"), load("ablation"), load("sensitivity")
    for n, d, k in [("methods", m, "method"), ("ablation", a, "variant"), ("sensitivity", s, ["param","value"])]:
        t = agg(d, k); t.to_csv(C.OUT_DIR / f"table_{n}.csv"); print(f"\n== {n}\n{t}")
    print("\nDBSCAN runtime (s):", m[m.method.str.startswith("DBSCAN")].runtime_s.mean().round(2))
