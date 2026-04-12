"""
track.py — Race track definition using waypoints.
The track is a closed loop defined by center points + width.
Cars must stay within the track boundaries.
"""
import math
import numpy as np


def make_oval_track(cx: float = 640, cy: float = 360, rx: float = 500, ry: float = 280,
                    n_points: int = 100, width: float = 80) -> dict:
    """Generate a simple oval track centered at (cx, cy)."""
    angles = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    centers = np.column_stack([
        cx + rx * np.cos(angles),
        cy + ry * np.sin(angles),
    ])

    # Compute normals at each point
    tangents = np.roll(centers, -1, axis=0) - centers
    norms = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    norms /= np.linalg.norm(norms, axis=1, keepdims=True)

    inner = centers - norms * (width / 2)
    outer = centers + norms * (width / 2)

    return {
        "centers": centers,
        "inner": inner,
        "outer": outer,
        "width": width,
        "n_points": n_points,
    }


def make_complex_track(width: float = 80) -> dict:
    """Generate a more interesting track with curves and straights."""
    # Define control points for the track center line
    control = np.array([
        [200, 600], [200, 200], [350, 100], [600, 100],
        [800, 150], [1000, 300], [1100, 500], [1000, 600],
        [800, 650], [600, 600], [400, 650],
    ], dtype=float)

    # Interpolate smooth curve through control points
    from scipy import interpolate
    n = len(control)
    t = np.arange(n + 1)
    # Close the loop
    control_closed = np.vstack([control, control[0]])

    cs_x = interpolate.CubicSpline(t, control_closed[:, 0], bc_type='periodic')
    cs_y = interpolate.CubicSpline(t, control_closed[:, 1], bc_type='periodic')

    t_fine = np.linspace(0, n, 200)
    centers = np.column_stack([cs_x(t_fine), cs_y(t_fine)])

    # Normals
    dx = cs_x(t_fine, 1)
    dy = cs_y(t_fine, 1)
    norms = np.column_stack([-dy, dx])
    norms /= np.linalg.norm(norms, axis=1, keepdims=True)

    inner = centers - norms * (width / 2)
    outer = centers + norms * (width / 2)

    return {
        "centers": centers,
        "inner": inner,
        "outer": outer,
        "width": width,
        "n_points": len(centers),
    }


def point_on_track(track: dict, pos: np.ndarray) -> tuple[bool, int, float]:
    """Check if a point is on the track.
    Returns (on_track, nearest_segment_idx, distance_from_center)."""
    diffs = track["centers"] - pos
    dists = np.linalg.norm(diffs, axis=1)
    idx = np.argmin(dists)
    dist = dists[idx]
    on_track = dist < track["width"] / 2
    return on_track, idx, dist


def progress_on_track(track: dict, segment_idx: int) -> float:
    """Return normalized progress [0, 1] along the track."""
    return segment_idx / track["n_points"]
