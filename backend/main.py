"""
FastAPI backend for the quantum UAV disaster-relief dashboard.

Runs the real QAOA optimization (~2-3s) as a background job so the
frontend can show live progress instead of blocking on a single request.
No faked progress -- the percentages and messages streamed to the frontend
come directly from inside the optimizer as it runs.
"""

import threading
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from optimizer import run_quantum_optimization, FLOOD_ZONES, UAV_BASES, build_distance_matrix

app = FastAPI(title="Quantum UAV Disaster Relief Dispatch")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store -- fine for a single-user local demo.
JOBS = {}


def _run_job(job_id: str):
    def progress_cb(frac, message):
        JOBS[job_id]["progress"] = frac
        JOBS[job_id]["message"] = message
        JOBS[job_id]["log"].append(message)

    try:
        result = run_quantum_optimization(progress_cb=progress_cb)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result
    except Exception as e:  # surfaced to the frontend rather than silently dying
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


@app.get("/api/scenario")
def get_scenario():
    """Static scenario info so the map can render immediately, before a run."""
    distance, uav_names, zone_names = build_distance_matrix(FLOOD_ZONES, UAV_BASES)
    return {
        "flood_zones": {name: {"lat": lat, "lon": lon} for name, (lat, lon) in FLOOD_ZONES.items()},
        "uav_bases": {name: {"lat": lat, "lon": lon} for name, (lat, lon) in UAV_BASES.items()},
        "distance_matrix": distance.tolist(),
        "uav_names": uav_names,
        "zone_names": zone_names,
    }


@app.post("/api/run")
def start_run():
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running", "progress": 0.0, "message": "Queued", "log": [], "result": None}
    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return {"status": "not_found"}
    return job


# Serve the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
