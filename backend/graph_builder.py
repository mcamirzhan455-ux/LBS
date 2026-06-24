"""
graph_builder.py

Builds the walking street graph used for local routing:
1. Downloads Vienna's OSM walk network
2. Attaches SRTM elevation to every node
3. Computes edge grades (slope)

Result is cached to disk as a pickle file.
"""

import logging
import os
import pickle

import osmnx as ox

ox.settings.max_query_area_size = 2_500_000_000

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
GRAPH_PATH = os.path.join(DATA_DIR, "vienna_walk_graph.pkl")
VIENNA_CENTER = (48.2082, 16.3738)  # Stephansdom
VIENNA_RADIUS_M = 8_000


def fetch_vienna_graph(force_refresh: bool = False):
    """Download (or load cached) Vienna walk network from OSM."""
    # FIX: use_cache=False prevents osmnx from reading its own stale cache,
    # but we still respect our own pickle cache below
    ox.settings.use_cache = False
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(GRAPH_PATH) and not force_refresh:
        logger.info("Found cached graph — loading from disk...")
        with open(GRAPH_PATH, "rb") as f:
            return pickle.load(f)

    logger.info("Downloading Vienna walk network from OpenStreetMap (1-2 min)...")
    # FIX: closed graph_from_point() call (was missing closing paren)
    G = ox.graph_from_point(
        center_point=VIENNA_CENTER,
        dist=VIENNA_RADIUS_M,
        network_type="walk",
        simplify=True,
    )
    logger.info(f"Downloaded {len(G.nodes):,} nodes and {len(G.edges):,} edges")

    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    logger.info(f"Saved to {GRAPH_PATH}")

    return G


def attach_elevations(G):
    """
    Read elevation for every node from a local DEM GeoTIFF (DEM_10m.tif).
    Falls back to 0 m for nodes outside the raster bounds.
    """
    import rasterio

    srtm_path = os.path.join(DATA_DIR, "srtm/DEM_10m.tif")

    # FIX: raise early with a helpful message if the DEM file is missing
    if not os.path.exists(srtm_path):
        raise FileNotFoundError(
            f"DEM file not found at {srtm_path}. "
            "Place DEM_10m.tif in data/srtm/."
        )

    logger.info("Reading elevations from local DEM file...")

    with rasterio.open(srtm_path) as src:
        elevation_data = src.read(1)
        height, width = elevation_data.shape

        for node_id, data in G.nodes(data=True):
            lat = data["y"]
            lon = data["x"]
            row, col = src.index(lon, lat)

            if 0 <= row < height and 0 <= col < width:
                elev = float(elevation_data[row, col])
                # FIX: SRTM nodata is typically -32768; treat it as 0
                G.nodes[node_id]["elevation"] = elev if elev > -9000 else 0.0
            else:
                G.nodes[node_id]["elevation"] = 0.0

    logger.info(f"Elevations attached to all {len(G.nodes):,} nodes.")
    return G


def compute_edge_grades(G):
    """
    Compute grade (rise/run) for every edge and store as 'grade' and 'grade_abs'.
    SRTM ~90 m resolution means short-edge diffs are noise — only record when
    the edge is long enough to be meaningful.
    """
    MIN_EDGE_LENGTH = 10  # metres — below this, grade is noise

    for u, v, data in G.edges(data=True):
        elev_u = G.nodes[u].get("elevation", 0.0)
        elev_v = G.nodes[v].get("elevation", 0.0)
        length = data.get("length", 1)

        elev_diff = elev_v - elev_u

        # FIX: skip grade calc on very short edges (SRTM noise)
        if length >= MIN_EDGE_LENGTH and abs(elev_diff) > 0:
            data["grade"] = elev_diff / length
        else:
            data["grade"] = 0.0

        data["grade_abs"] = abs(data["grade"])

    return G


def build_graph(force_refresh: bool = False):
    """Run the full pipeline: download → elevations → grades → save."""
    logger.info("=== Starting graph build pipeline ===")

    G = fetch_vienna_graph(force_refresh=force_refresh)
    G = attach_elevations(G)
    G = compute_edge_grades(G)

    enriched_path = GRAPH_PATH.replace(".pkl", "_enriched.pkl")
    with open(enriched_path, "wb") as f:
        pickle.dump(G, f)

    logger.info(f"=== Done! Enriched graph saved to {enriched_path} ===")
    return G


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_graph()
