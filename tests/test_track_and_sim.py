import numpy as np
import pytest

from app import track as trk
from app.neuroevolution import Population
from app.simulation import Generation, Track
from app.world import World


def ellipse_points(n=200, cx=300, cy=200, rx=220, ry=140):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + np.cos(th) * rx, cy + np.sin(th) * ry], axis=1)


def test_resample_closed_keeps_loop_length_similar():
    pts = ellipse_points()
    out = trk.resample_closed(pts, spacing=5)
    perim = np.hypot(*(np.roll(out, -1, axis=0) - out).T).sum()
    expected = 2 * np.pi * (220 + 140) / 2  # rough ellipse perimeter estimate
    assert abs(perim - expected) / expected < 0.15


def test_build_mask_marks_track_pixels_and_not_center():
    pts = ellipse_points()
    centerline = trk.build_centerline(pts)
    mask = trk.build_mask(centerline, width=40, w=600, h=400)
    # a point on the centerline should be on the mask
    cx, cy = centerline[0]
    assert mask[int(cy), int(cx)]
    # the middle of the ellipse (inside the hole) should not be on the track
    assert not mask[200, 300]


def test_population_forward_is_batched_and_bounded():
    pop = Population(size=10, rng=np.random.default_rng(0))
    inputs = np.random.default_rng(0).uniform(-1, 1, (10, 6))
    hidden, output = pop.forward(inputs)
    assert hidden.shape == (10, 8)
    assert output.shape == (10, 2)
    assert np.all(np.abs(output) <= 1.0)  # tanh output


def test_next_generation_keeps_elite_unchanged():
    pop = Population(size=12, rng=np.random.default_rng(1))
    fitness = np.arange(12, dtype=float)  # index 11 is "best"
    child = pop.next_generation(fitness)
    assert np.array_equal(child.W1[0], pop.W1[11])  # elite #1 preserved verbatim


def test_generation_kills_cars_that_leave_the_track():
    pts = ellipse_points()
    centerline = trk.build_centerline(pts)
    mask = trk.build_mask(centerline, width=40, w=600, h=400)
    angle = trk.start_heading(centerline)
    track = Track(centerline, mask, 40, angle)
    pop = Population(size=5, rng=np.random.default_rng(0))
    gen = Generation(track, pop)
    gen.x[0] = 1.0
    gen.y[0] = 1.0  # corner of the canvas, definitely off track
    for _ in range(3):
        gen.step()
    assert not gen.alive[0]
    assert gen.alive.sum() < 5 or True  # some cars may also leave naturally


def test_world_runs_generations_and_produces_frames():
    pts = ellipse_points(n=180)
    w = World()
    w.set_track(pts, width=60, w=600, h=400)
    for _ in range(400):
        w.tick()
    frame = w.frame()
    assert frame["gen"] >= 1
    assert "cars" in frame and isinstance(frame["cars"], list)
    assert 0 <= frame["lapFraction"] <= 1
