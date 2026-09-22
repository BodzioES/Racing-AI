"""Turns a hand-drawn polyline into a closed, smooth centerline plus a
pixel mask that the simulation uses for collision and ray-casting.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .config import CENTERLINE_SPACING


def resample_closed(points: np.ndarray, spacing: float) -> np.ndarray:
    """Re-parametrize a closed polyline to evenly spaced points, ~spacing px apart."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    nxt = np.roll(pts, -1, axis=0)
    seg_len = np.hypot(*(nxt - pts).T)
    total = seg_len.sum()
    count = max(24, round(total / spacing))
    step = total / count
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])[:-1]

    out = np.empty((count, 2))
    seg_i = 0
    for k in range(count):
        d = k * step
        while seg_i < n - 1 and cum[seg_i] + seg_len[seg_i] < d:
            seg_i += 1
        denom = seg_len[seg_i]
        tt = (d - cum[seg_i]) / denom if denom > 0 else 0.0
        out[k] = pts[seg_i] + (nxt[seg_i] - pts[seg_i]) * tt
    return out


def smooth_closed(points: np.ndarray, radius: int, iterations: int) -> np.ndarray:
    """Moving-average smoothing on a closed loop (wraps around)."""
    cur = np.asarray(points, dtype=np.float64)
    n = len(cur)
    for _ in range(iterations):
        nxt = np.zeros_like(cur)
        for offset in range(-radius, radius + 1):
            nxt += np.roll(cur, offset, axis=0)
        cur = nxt / (2 * radius + 1)
    return cur


def build_centerline(raw_points, spacing: float = CENTERLINE_SPACING) -> np.ndarray:
    """Raw hand-drawn points -> clean, evenly spaced, smoothed closed loop."""
    res = resample_closed(np.asarray(raw_points, dtype=np.float64), spacing)
    smoothed = smooth_closed(res, radius=3, iterations=3)
    return resample_closed(smoothed, spacing)


def build_mask(centerline: np.ndarray, width: float, w: int, h: int) -> np.ndarray:
    """Rasterize a thick closed line into a boolean (h, w) drivable-surface mask."""
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    loop = [tuple(p) for p in centerline] + [tuple(centerline[0])]
    draw.line(loop, fill=255, width=max(1, round(width)), joint="curve")
    r = width / 2
    for p in (centerline[0], centerline[-1]):
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)
    return np.asarray(img, dtype=bool)


def start_heading(centerline: np.ndarray) -> float:
    """Heading (radians) a car spawned at index 0 should face."""
    dx = centerline[1][0] - centerline[0][0]
    dy = centerline[1][1] - centerline[0][1]
    return float(np.arctan2(dy, dx))
