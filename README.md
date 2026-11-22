# SIH26008 – DBSCAN Anomaly Clustering for Conveyor Belt Joint Rupture Prediction

## Project layout
```
conveyor_dbscan/
├── config.py            # ALL tunable settings (DBSCAN, windows, alert rules)
├── simulate_data.py     # synthetic sensor + incident data generator
├── run_pipeline.py      # features -> DBSCAN -> alerts -> validation -> plots
├── requirements.txt
├── data/                # sensor_readings.csv, rupture_incidents.csv
└── outputs/             # results (created on run)
```

## Step-by-step: run it
1. Install Python 3.9+ and open a terminal in the project folder.
2. (Recommended) create a virtual environment:
   - Windows: `python -m venv venv && venv\Scripts\activate`
   - Linux/Mac: `python3 -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python run_pipeline.py --generate`   (first run: creates synthetic data, then analyses it)
5. Later runs: `python run_pipeline.py`

## Outputs (in `outputs/`)
| File | Meaning |
|---|---|
| `summary.json` | detection rate, lead time, false alarms, precision/recall |
| `incident_validation.csv` | each logged rupture: detected? first alert time, lead time (h) |
| `alerts.csv` | all alerts raised (with `true_alert` = matched a rupture) |
| `labeled_readings.csv` | every reading with DBSCAN `cluster` (-1 = noise) and `flag` |
| `k_distance.png` | eps selection curve |
| `clusters_pca.png` | normal clusters vs. noise points |
| `timeline_<joint>.png` | sensor traces, flagged points in red, rupture line |

## How it works
1. **Collect** – CSV with `timestamp, joint_id, vibration_rms, tension_kN, temperature_C`.
2. **Features** – 3 raw readings + 1-hour rolling mean + rolling std per joint (9 features), scaled with `RobustScaler`.
3. **DBSCAN** – `eps` auto-picked from the k-distance knee (or set manually). Big dense clusters = normal operating
   regimes (load cycles). Noise (-1) and tiny clusters (<1% of data) = anomalous.
4. **Alerts** – flagged points on one joint within 60 min merge; an alert needs >= 12 points, so one-off sensor
   glitches are ignored.
5. **Validation** – each alert is checked against `rupture_incidents.csv`: an alert within `LOOKBACK_HOURS`
   (48 h) before a rupture on that joint is a true positive; others are false alarms. Lead time = rupture time − first alert.

Sample result on the synthetic data: 4/4 ruptures detected, ~17 h mean lead time, 1 false alarm.
(Point-level recall is lower by design: the 48 h window includes hours before the degradation has started.)

## How to make changes
| I want to… | Do this |
|---|---|
| Use **real data** | Replace the two CSVs in `data/` (same column names). Run `python run_pipeline.py` (no `--generate`). Change `SENSOR_COLS` in `config.py` to add sensors. |
| More/fewer anomalies flagged | `python run_pipeline.py --eps 1.0` (lower = more flags) or change `EPS_SCALE` in `config.py` |
| Change density requirement | `--min-samples 30` or `MIN_SAMPLES` (keep >= 2 × features) |
| Change sampling rate | Adjust `ROLL_WINDOW` (samples per hour) and `SAMPLE_FREQ` |
| Fewer false alarms / earlier warnings | Raise `MIN_ALERT_POINTS` (fewer false alarms) or lower it (earlier alerts) |
| Change validation window | `--lookback-hours 72` or `LOOKBACK_HOURS` |
| New features (e.g. FFT, rate of change) | Edit `build_features()` in `run_pipeline.py` |
| Different data scale (more joints/days) | `N_JOINTS`, `N_DAYS` in `config.py`, then `--generate` |

## Tips
- Tune `eps` by looking at `k_distance.png`: pick the "elbow".
- For a live system, run the pipeline on a rolling window of recent data and push `alerts.csv` to a dashboard/SMS.
- Add a Streamlit dashboard on top of `labeled_readings.csv` as an extension.

## Reproducing the IEEE report results
```
python run_pipeline.py --generate          # single-run walkthrough (seed 42) -> outputs/
python experiments.py 0 4                  # multi-seed benchmark in chunks (each call < 5 min)
python experiments.py 4 8
python experiments.py 8 10
python aggregate_experiments.py            # -> outputs/table_methods.csv, table_ablation.csv, table_sensitivity.csv
python experiments_weak.py 0 5             # weak-precursor stress test (precursor x0.3)
python experiments_weak.py 5 10
python make_report_figures.py              # -> report_figs/*.png used in the report
```
Set `PRECURSOR_SCALE` in `config.py` to change precursor strength. Replace the CSVs in `data/` with real plant data
and re-run `run_pipeline.py` to validate on real incidents.

## Frontend (web dashboard)
```
pip install -r requirements.txt
uvicorn api:app --reload          # then open http://127.0.0.1:8000
```
- `api.py` = FastAPI backend (wraps run_pipeline.py). Interactive API docs: http://127.0.0.1:8000/docs
- `frontend/index.html` = dashboard (Chart.js). Change eps / min_samples / lookback and click "Run analysis".
- Endpoints: `POST /api/run` (JSON: eps, min_samples, lookback_hours) and `GET /api/joint/{joint_id}`.
- Separate React/Vue/Vite frontend: call the same endpoints (set `API = "http://127.0.0.1:8000"`); CORS is already enabled.
- Real data: replace the CSVs in `data/`, restart uvicorn.
