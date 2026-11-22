"""Central settings. Edit values here (or override eps/min_samples on the command line)."""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

SENSOR_CSV = DATA_DIR / "sensor_readings.csv"      # timestamp, joint_id, vibration_rms, tension_kN, temperature_C
INCIDENT_CSV = DATA_DIR / "rupture_incidents.csv"  # incident_id, joint_id, rupture_time

SENSOR_COLS = ["vibration_rms", "tension_kN", "temperature_C"]

# ---- Synthetic data generator ----
SEED = 42
N_JOINTS = 10
N_DAYS = 10
SAMPLE_FREQ = "10min"
N_RUPTURES = 4
PRECURSOR_HOURS = 36        # how long before rupture the degradation begins
PRECURSOR_SCALE = 1.0       # 1.0 = strong precursor; 0.3 = weak/subtle precursor (stress test)
GLITCH_RATE = 0.003         # random one-off sensor glitches (should NOT become alerts)

# ---- Feature engineering ----
ROLL_WINDOW = 6             # samples (6 x 10 min = 1 hour)

# ---- DBSCAN ----
MIN_SAMPLES = 20            # rule of thumb: >= 2 x number of features
EPS = None                  # None = auto-select from k-distance knee
EPS_SCALE = 1.0             # multiply auto eps (raise for fewer flags, lower for more)
MIN_CLUSTER_FRAC = 0.01     # clusters smaller than 1% of data are treated as anomalous

# ---- Alerting & validation ----
ALERT_GAP_MIN = 60          # flagged points within 60 min merge into one alert
MIN_ALERT_POINTS = 12       # persistent anomaly (>= 2 h of flags); rolling window smears a glitch over ~6 points
LOOKBACK_HOURS = 48         # window before a rupture in which alerts count as true positives
