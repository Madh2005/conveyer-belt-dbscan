"""Generate synthetic vibration / tension / temperature data plus rupture incident logs.
Replace the CSVs in data/ with real data of the same columns to use the pipeline on real belts."""
import numpy as np, pandas as pd
import config as C


def simulate():
    rng = np.random.default_rng(C.SEED)
    idx = pd.date_range("2026-01-01", periods=int(pd.Timedelta(days=C.N_DAYS) / pd.Timedelta(C.SAMPLE_FREQ)),
                        freq=C.SAMPLE_FREQ)
    hours = (idx - idx[0]).total_seconds().values / 3600
    load = 0.5 + 0.5 * np.sin(2 * np.pi * hours / 24 - 1.0)       # daily load cycle
    ambient = 3 * np.sin(2 * np.pi * hours / 24 - 2.0)

    rupture_joints = rng.choice(C.N_JOINTS, C.N_RUPTURES, replace=False)
    rupture_times = {j: idx[int(len(idx) * rng.uniform(0.65, 0.92))] for j in rupture_joints}

    frames, incidents = [], []
    for j in range(C.N_JOINTS):
        vib = 2.0 + 0.1 * j + 0.6 * load + rng.normal(0, 0.15, len(idx))
        ten = 48 + 0.5 * j + 8 * load + rng.normal(0, 1.2, len(idx))
        tmp = 38 + 4 * load + ambient + rng.normal(0, 0.6, len(idx))

        if j in rupture_times:
            r = rupture_times[j]
            p = np.clip((idx - (r - pd.Timedelta(hours=C.PRECURSOR_HOURS))) / pd.Timedelta(hours=C.PRECURSOR_HOURS), 0, 1) ** 1.5
            k = C.PRECURSOR_SCALE
            vib += k * 3.5 * p + k * rng.normal(0, 0.4, len(idx)) * p
            ten += k * 12 * p + k * rng.normal(0, 3.0, len(idx)) * p
            tmp += k * 14 * p + k * rng.normal(0, 1.0, len(idx)) * p
            incidents.append({"incident_id": f"INC-{len(incidents)+1:03d}", "joint_id": f"J{j+1:02d}",
                              "rupture_time": r})

        # random one-off glitches
        g = rng.random(len(idx)) < C.GLITCH_RATE
        vib[g] += rng.uniform(2, 4, g.sum()); ten[g] -= rng.uniform(10, 20, g.sum())

        df = pd.DataFrame({"timestamp": idx, "joint_id": f"J{j+1:02d}",
                           "vibration_rms": vib.round(3), "tension_kN": ten.round(2), "temperature_C": tmp.round(2)})
        if j in rupture_times:                       # belt stops after rupture
            df = df[df.timestamp <= rupture_times[j]]
        frames.append(df)

    C.DATA_DIR.mkdir(exist_ok=True)
    pd.concat(frames).to_csv(C.SENSOR_CSV, index=False)
    pd.DataFrame(incidents).sort_values("rupture_time").to_csv(C.INCIDENT_CSV, index=False)
    print(f"Wrote {C.SENSOR_CSV} and {C.INCIDENT_CSV}")


if __name__ == "__main__":
    simulate()
