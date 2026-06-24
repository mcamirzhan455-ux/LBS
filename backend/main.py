"""
main.py

FastAPI web server for the Vienna Elevation Router.
Exposes:
  GET  /        — serves the frontend HTML
  GET  /health  — liveness check
  POST /routes  — returns 3 elevation-aware route options
"""

import logging
import os
import pickle

from fastapi.staticfiles import StaticFiles

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from backend.graph_builder import build_graph
from backend.router import compute_routes as compute_routes_local
from backend.router_gh import compute_routes as compute_routes_gh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
ENRICHED_GRAPH_PATH = os.path.join(DATA_DIR, "vienna_walk_graph_enriched.pkl")
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "../frontend/index.html")

VIENNA_BOUNDS = {"lat": (48.10, 48.33), "lon": (16.18, 16.58)}

G = None

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event("startup"))
# FIX: on_event("startup") is deprecated since FastAPI 0.93 — use lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global G
    use_gh = os.getenv("USE_GRAPHHOPPER", "").strip() == "1"
    if use_gh:
        logger.info("GraphHopper mode — skipping local graph load.")
    elif os.path.exists(ENRICHED_GRAPH_PATH):
        logger.info("Loading enriched graph from disk...")
        with open(ENRICHED_GRAPH_PATH, "rb") as f:
            G = pickle.load(f)
        logger.info(f"Graph loaded: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
    else:
        logger.info("No cached graph found — building from scratch (may take a few minutes)...")
        G = build_graph()
    yield
    # cleanup on shutdown (nothing needed here)


app = FastAPI(
    title="Vienna Elevation Router",
    description="Returns 3 elevation-aware walking routes between two points in Vienna",
    version="1.1",
    lifespan=lifespan,
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "../frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_vienna(lat: float, lon: float) -> bool:
    return (VIENNA_BOUNDS["lat"][0] <= lat <= VIENNA_BOUNDS["lat"][1] and
            VIENNA_BOUNDS["lon"][0] <= lon <= VIENNA_BOUNDS["lon"][1])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    origin_lat: float
    origin_lon: float
    dest_lat: float | None = None
    dest_lon: float | None = None
    router: str = "local"       # "local" | "graphhopper"
    mode: str = "point_to_point"  # "point_to_point" | "loop"
    park_name: str | None = None
    loop_distance_km: float = 5.0

    # FIX: validate router and mode values so bad input fails fast with a clear message
    @field_validator("router")
    @classmethod
    def validate_router(cls, v):
        if v not in ("local", "graphhopper"):
            raise ValueError("router must be 'local' or 'graphhopper'")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        if v not in ("point_to_point", "loop", "park_loop"):
            raise ValueError("mode must be 'point_to_point', 'loop' or 'park_loop'")
        return v

    # FIX: clamp loop distance to a sane range
    @field_validator("loop_distance_km")
    @classmethod
    def validate_loop_distance(cls, v):
        if not (0.5 <= v <= 50.0):
            raise ValueError("loop_distance_km must be between 0.5 and 50")
        return v


class ElevationPoint(BaseModel):
    distance: float
    elevation: float


class RouteResult(BaseModel):
    profile: str
    distance_m: float
    elevation_gain_m: float
    elevation_loss_m: float
    coordinates: list[list[float]]
    elevation_profile: list[ElevationPoint]


class RouteResponse(BaseModel):
    routes: list[RouteResult]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def serve_frontend():
    if not os.path.exists(FRONTEND_PATH):
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(FRONTEND_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "graph_loaded": G is not None,
        "node_count": len(G.nodes) if G else 0,
    }

@app.get("/parks")
def get_parks():
    from backend.router_gh import list_parks
    return {"parks": list_parks()}



@app.post("/routes", response_model=RouteResponse)
def get_routes(req: RouteRequest):
    use_gh = req.router == "graphhopper"

    if not use_gh and G is None:
        raise HTTPException(status_code=503, detail="Graph not loaded yet — please wait")

    if not _in_vienna(req.origin_lat, req.origin_lon):
        raise HTTPException(status_code=400, detail="Origin coordinates are outside Vienna")

    if req.mode == "point_to_point":
        if req.dest_lat is None or req.dest_lon is None:
            raise HTTPException(status_code=400, detail="Destination required for point_to_point mode")
        if not _in_vienna(req.dest_lat, req.dest_lon):
            raise HTTPException(status_code=400, detail="Destination coordinates are outside Vienna")

    fn = compute_routes_gh if use_gh else compute_routes_local

    try:
        routes = fn(
            G,
            req.origin_lat, req.origin_lon,
            req.dest_lat, req.dest_lon,
            mode=req.mode,
            loop_distance_km=req.loop_distance_km,
            park_name=req.park_name,
        )
    except RuntimeError as e:
        # FIX: propagate GH config errors as 500 with a readable message
        raise HTTPException(status_code=500, detail=str(e))

    if not routes:
        raise HTTPException(status_code=404, detail="No routes found between these points")

    return RouteResponse(routes=routes)
