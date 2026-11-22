"""REST API wrapping the DBSCAN pipeline.  Run:  uvicorn api:app --reload   then open http://127.0.0.1:8000"""
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import config as C, run_pipeline as P

app = FastAPI(title="Conveyor Belt DBSCAN API")
# CORS: lets a separate frontend (React dev server on :3000/:5173) call this API. Restrict origins in production.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATE = {}


class RunParams(BaseModel):
    eps: Optional[float] = None            # None = automatic (k-distance knee)
    min_samples: int = C.MIN_SAMPLES
    lookback_hours: int = C.LOOKBACK_HOURS


def records(df):
    return json.loads(df.to_json(orient="records", date_format="iso"))


@app.post("/api/run")
def run(p: RunParams):
    """Run the full pipeline and return summary, per-incident validation and alerts."""
    if not C.SENSOR_CSV.exists():
        import simulate_data; simulate_data.simulate()
    df, inc = P.load()
    df, X, kd, eps = P.run_dbscan(df, p.eps, p.min_samples)
    alerts = P.make_alerts(df)
    res, alerts, summary = P.validate(df, alerts, inc, p.lookback_hours)
    summary.update(eps=round(eps, 3), min_samples=p.min_samples, flagged_points=int(df.flag.sum()), total_points=int(len(df)))
    STATE.update(df=df, inc=inc)
    return {"summary": summary, "incidents": records(res), "alerts": records(alerts),
            "joints": sorted(df.joint_id.unique().tolist())}


@app.get("/api/joint/{joint_id}")
def joint(joint_id: str):
    """Time series of one joint with DBSCAN flags (run /api/run first)."""
    if "df" not in STATE:
        raise HTTPException(409, "Run the pipeline first: POST /api/run")
    g = STATE["df"].query("joint_id == @joint_id")
    if g.empty:
        raise HTTPException(404, "Unknown joint")
    r = STATE["inc"].query("joint_id == @joint_id")
    return {"t": g.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            "vibration": g.vibration_rms.tolist(), "tension": g.tension_kN.tolist(),
            "temperature": g.temperature_C.tolist(), "flag": g.flag.astype(int).tolist(),
            "rupture": r.rupture_time.iloc[0].strftime("%Y-%m-%dT%H:%M:%S") if len(r) else None}


@app.get("/")
def index():
    return FileResponse(C.BASE_DIR / "frontend" / "index.html")
