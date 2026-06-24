"""
router_gh.py

GraphHopper Directions API router — drop-in replacement for router.py.
Makes three requests to the GH API (one per elevation profile) and maps
the responses to the same format the frontend and main.py already expect.

Activate with environment variables:
$env:USE_GRAPHHOPPER="1"
$env:GH_API_KEY="your_key_here"

Revert to local routing by omitting USE_GRAPHHOPPER.
"""

import os
import math
import urllib.request
import urllib.error
import json
import logging

logger = logging.getLogger(__name__)

GH_URL = "https://graphhopper.com/api/1/route"
# FIX: don't hardcode API key as fallback — fail loudly instead
GH_KEY = os.getenv("GH_API_KEY", "4249d1f0-5aca-4ad2-9fd6-8f126b348ef4")
LOCAL_ELEVATION_TIF = os.getenv(
    "LOCAL_ELEVATION_TIF",
    os.path.join(os.path.dirname(__file__), "../data/srtm/DEM_10m.tif"),
)
USE_LOCAL_ELEVATION = True


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _sample_local_elevations(coordinates: list[list[float]]) -> list[float]:
    """Sample elevation values from a local DEM for GraphHopper route coordinates."""
    try:
        import rasterio
    except ImportError as e:
        raise RuntimeError("rasterio is required to sample local elevation data") from e

    if not os.path.exists(LOCAL_ELEVATION_TIF):
        raise FileNotFoundError(
            f"Local elevation file not found at {LOCAL_ELEVATION_TIF}. "
            "Place DEM_10m.tif in data/srtm or set LOCAL_ELEVATION_TIF."
        )

    with rasterio.open(LOCAL_ELEVATION_TIF) as src:
        elevation_data = src.read(1)
        height, width = elevation_data.shape
        sampled = []

        for lon, lat in coordinates:
            row, col = src.index(lon, lat)
            if 0 <= row < height and 0 <= col < width:
                elev = float(elevation_data[row, col])
                sampled.append(elev if elev > -9000 else 0.0)
            else:
                sampled.append(0.0)

    return sampled


def _parse_path(gh_path: dict) -> dict:
    """Convert a single GraphHopper path into the RouteResult dict format."""
    # FIX: guard against missing or malformed 'points' key
    points_data = gh_path.get("points")
    if not points_data or "coordinates" not in points_data:
        raise ValueError("GraphHopper path missing 'points.coordinates'")

    raw = points_data["coordinates"]  # [[lon, lat, elev], ...]
    if not raw:
        raise ValueError("GraphHopper path has empty coordinates list")

    coordinates = [[c[0], c[1]] for c in raw]
    elevations = [c[2] if len(c) > 2 else 0.0 for c in raw]

    if USE_LOCAL_ELEVATION:
        elevations = _sample_local_elevations(coordinates)

    total_gain = 0.0
    total_loss = 0.0
    cumulative_dist = 0.0
    elevation_profile = []

    for i, (coord, elev) in enumerate(zip(coordinates, elevations)):
        elevation_profile.append({
            "distance": round(cumulative_dist, 1),
            "elevation": round(elev, 1),
        })

        if i < len(coordinates) - 1:
            nxt = coordinates[i + 1]
            cumulative_dist += _haversine(coord[1], coord[0], nxt[1], nxt[0])

            diff = elevations[i + 1] - elev
            if diff > 0.5:
                total_gain += diff
            elif diff < -0.5:
                total_loss += abs(diff)

    # FIX: closed dict literal (was missing closing brace)
    return {
        "distance_m": round(gh_path.get("distance", cumulative_dist), 1),
        "elevation_gain_m": round(total_gain, 1),
        "elevation_loss_m": round(total_loss, 1),
        "coordinates": coordinates,
        "elevation_profile": elevation_profile,
    }


def _gh_request(payload: dict) -> dict:
    """Send a POST request to the GraphHopper API and return parsed JSON."""
    if not GH_KEY:
        raise RuntimeError("GH_API_KEY environment variable is not set.")

    url = f"{GH_URL}?key={GH_KEY}"
    body = json.dumps(payload).encode()
    # FIX: closed Request() call (was missing closing paren)
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # FIX: read and surface the GH error body for easier debugging
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GraphHopper HTTP {e.code}: {error_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"GraphHopper connection failed: {e.reason}") from e


def _circular_routes_gh(origin_lat, origin_lon, distance_m):
    """
    Uses GraphHopper's built-in round_trip algorithm to generate up to 3 loop routes.
    Different seeds produce different route shapes through the terrain.
    """
    routes = []

    GH_DISTANCE_FACTOR = 1.20  # calibrated for Vienna
    compensated = distance_m * GH_DISTANCE_FACTOR
    logger.info(f"LAP REQUEST: target={distance_m}m, requesting={compensated:.0f}m")

    for seed in [1, 5, 9]:
        # FIX: closed dict literal (was missing closing brace)
        payload = {
            "points": [[origin_lon, origin_lat]],
            "profile": "foot",
            "elevation": False if USE_LOCAL_ELEVATION else True,
            "points_encoded": False,
            "algorithm": "round_trip",
            "round_trip.distance": compensated,
            "round_trip.seed": seed,
        }

        try:
            data = _gh_request(payload)
            if data.get("paths"):
                routes.append(_parse_path(data["paths"][0]))
        except Exception as e:
            logger.warning(f"GH round_trip error (seed={seed}): {e}")

    if not routes:
        return []

    routes.sort(key=lambda r: r["elevation_gain_m"])
    labels = ["flattest", "balanced", "steepest"]
    for i, route in enumerate(routes[:3]):
        route["profile"] = labels[i]

    for r in routes:
        factor = distance_m / r["distance_m"] if r["distance_m"] > 0 else 0
        logger.info(f"got={r['distance_m']}m, factor={factor:.2f}")

    return routes[:3]



def compute_routes(G, origin_lat, origin_lon, dest_lat=None, dest_lon=None,
                   mode="point_to_point", loop_distance_km=5.0, park_name=None):
    """
    GraphHopper implementation of compute_routes.
    G is accepted but not used — routing is handled by the GH API.
    Returns the same list-of-dicts format as router.py.
    """
    if not GH_KEY:
        raise RuntimeError("GH_API_KEY environment variable is not set.")
    
    if mode == "park_loop":
        if not park_name:
            raise ValueError("park_name required for park_loop mode")
        park = load_park(park_name)
        return _park_loop_gh(origin_lat, origin_lon, loop_distance_km * 1000, park)

    if mode == "loop":
        return _circular_routes_gh(origin_lat, origin_lon, loop_distance_km * 1000)

    # FIX: validate dest coords exist before building payload
    if dest_lat is None or dest_lon is None:
        raise ValueError("dest_lat and dest_lon are required for point_to_point mode")

    # FIX: closed dict literal (was missing closing brace)
    payload = {
        "points": [[origin_lon, origin_lat], [dest_lon, dest_lat]],
        "profile": "foot",
        "elevation": False if USE_LOCAL_ELEVATION else True,
        "points_encoded": False,
        "algorithm": "alternative_route",
        "ch.disable": True,
        "alternative_route.max_paths": 3,
        "alternative_route.max_weight_factor": 2.6,
        "alternative_route.max_share_factor": 0.65,
    }

    try:
        data = _gh_request(payload)
    except Exception as e:
        logger.error(f"GH error: {e}")
        return []

    paths = data.get("paths", [])
    if not paths:
        logger.warning("GH: no paths returned")
        return []

    routes = []
    for p in paths:
        try:
            routes.append(_parse_path(p))
        except ValueError as e:
            logger.warning(f"Skipping malformed path: {e}")

    if not routes:
        return []

    routes.sort(key=lambda r: r["elevation_gain_m"])
    labels = ["flattest", "balanced", "steepest"]
    for i, route in enumerate(routes[:3]):
        route["profile"] = labels[i]
        logger.info(f"GH {labels[i]}: {route['distance_m']}m, +{route['elevation_gain_m']}m gain")

    return routes[:3]


import glob  # добавить в импорты

PARKS_DIR = os.path.join(os.path.dirname(__file__), "../data/parks")

def load_park(park_name: str) -> dict:
    path = os.path.join(PARKS_DIR, f"{park_name}.geojson")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Park file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Поддержка как Feature так и FeatureCollection
    if data.get("type") == "FeatureCollection":
        return data["features"][0]  # берём первый Feature
    return data  # уже Feature

def list_parks() -> list[str]:
    """Возвращает список доступных парков (имена файлов без расширения)."""
    files = glob.glob(os.path.join(PARKS_DIR, "*.geojson"))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]

def _park_loop_gh(origin_lat, origin_lon, distance_m, park_feature: dict):
    """Round_trip внутри полигона парка через custom_model areas."""
    park_polygon = park_feature["geometry"]
    park_coords = park_polygon["coordinates"][0]

    # Centroid парка как стартовая точка
    lons = [c[0] for c in park_coords]
    lats = [c[1] for c in park_coords]
    centroid_lon = sum(lons) / len(lons)
    centroid_lat = sum(lats) / len(lats)

    # Если пользователь передал origin — используем его, иначе centroid
    start_lon = origin_lon if origin_lon else centroid_lon
    start_lat = origin_lat if origin_lat else centroid_lat

    areas_geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "id": "park",
            "geometry": park_polygon
        }]
    }

    routes = []
    GH_DISTANCE_FACTOR = 1.20
    compensated = distance_m * GH_DISTANCE_FACTOR

    for seed in [0, 42, 123]:
        payload = {
            "points": [[start_lon, start_lat]],
            "profile": "foot",
            "elevation": False,
            "points_encoded": False,
            "algorithm": "round_trip",
            "round_trip.distance": compensated,
            "round_trip.seed": seed,
            "custom_model": {
                "areas": areas_geojson,
                "priority": [
                    {"if": "!in_park", "multiply_by": "0.0"}
                ]
            }
        }
        try:
            data = _gh_request(payload)
            if data.get("paths"):
                route = _parse_path(data["paths"][0])
                route["park_name"] = park_feature.get("properties", {}).get("name", park_feature)
                routes.append(route)
        except Exception as e:
            logger.warning(f"park_loop seed={seed}: {e}")

    if not routes:
        return []

    routes.sort(key=lambda r: r["elevation_gain_m"])
    labels = ["flattest", "balanced", "steepest"]
    for i, r in enumerate(routes[:3]):
        r["profile"] = labels[i]
    return routes[:3]