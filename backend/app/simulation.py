"""The racetrack world: car physics, ray-sensors and progress tracking.

Every car is a row in a set of NumPy arrays. A "step" advances the whole
population at once -- there is no per-car Python loop in the hot path,
which is what lets this run fast enough for 80 cars at up to 20x speed.
"""
from __future__ import annotations

import math

import numpy as np

from .config import (
    LAPS_TO_MASTER,
    MAX_SPEED,
    MIN_SPEED,
    POP,
    RAY_ANGLES,
    RAY_STEP,
    SEARCH_WINDOW,
    STALL_STEPS,
    TURN_RATE,
)
from .neuroevolution import Population

_SEARCH_OFFSETS = np.array(list(SEARCH_WINDOW), dtype=np.int64)


class Track:
    def __init__(self, centerline: np.ndarray, mask: np.ndarray, width: float, start_angle: float):
        self.centerline = centerline          # (N, 2)
        self.mask = mask                      # (h, w) bool
        self.width = width
        self.start_angle = start_angle
        self.n = len(centerline)
        self.h, self.w = mask.shape
        ray_len = max(110.0, width * 1.9)
        self.ray_n = max(1, math.ceil(ray_len / RAY_STEP))
        self.car_length = min(20.0, max(11.0, width * 0.26))

    def on_track(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xi = x.astype(np.int64)
        yi = y.astype(np.int64)
        inb = (xi >= 0) & (yi >= 0) & (xi < self.w) & (yi < self.h)
        out = np.zeros(x.shape, dtype=bool)
        out[inb] = self.mask[yi[inb], xi[inb]]
        return out


class Generation:
    """One population's run on one track, from spawn to generation end."""

    def __init__(self, track: Track, population: Population):
        self.track = track
        self.pop = population
        n = population.size
        self.x = np.full(n, track.centerline[0, 0])
        self.y = np.full(n, track.centerline[0, 1])
        self.angle = np.full(n, track.start_angle)
        self.speed = np.full(n, MIN_SPEED + 1.0)
        self.alive = np.ones(n, dtype=bool)
        self.idx = np.zeros(n, dtype=np.int64)
        self.progress = np.zeros(n)
        self.best = np.zeros(n)
        self.last_improve = np.zeros(n, dtype=np.int64)
        self.steps = np.zeros(n, dtype=np.int64)
        self.first_lap = np.full(n, -1, dtype=np.int64)
        self.rays = np.ones((n, 5))
        self.step_count = 0
        self.gen_best = 0.0
        self.death_marks: list[tuple[float, float]] = []

    # -- sensing -----------------------------------------------------
    def _cast_rays(self) -> np.ndarray:
        """5 ray distances (0..1, fraction of ray length) for every car."""
        track = self.track
        angles = self.angle[:, None] + RAY_ANGLES[None, :]     # (n, 5)
        dx = np.cos(angles) * RAY_STEP
        dy = np.sin(angles) * RAY_STEP
        x = np.repeat(self.x[:, None], 5, axis=1)
        y = np.repeat(self.y[:, None], 5, axis=1)
        hit_step = np.full(angles.shape, track.ray_n, dtype=np.int64)
        found = np.zeros(angles.shape, dtype=bool)
        for k in range(1, track.ray_n + 1):
            x = x + dx
            y = y + dy
            off = ~track.on_track(x, y)
            newly = off & ~found
            hit_step[newly] = k - 1
            found |= off
        return hit_step / track.ray_n

    # -- one physics + learning step for the whole population --------
    def step(self) -> None:
        track = self.track
        alive = self.alive
        if not alive.any():
            return
        self.rays[alive] = self._cast_rays()[alive]

        inputs = np.empty((self.pop.size, 6))
        inputs[:, :5] = self.rays
        inputs[:, 5] = self.speed / MAX_SPEED
        _, out = self.pop.forward(inputs)
        steer, throttle = out[:, 0], out[:, 1]

        target = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * (throttle * 0.5 + 0.5)
        self.speed = np.where(alive, self.speed + (target - self.speed) * 0.12, self.speed)
        self.angle = np.where(
            alive, self.angle + steer * TURN_RATE * (0.3 + 0.7 * self.speed / MAX_SPEED), self.angle
        )
        self.x = np.where(alive, self.x + np.cos(self.angle) * self.speed, self.x)
        self.y = np.where(alive, self.y + np.sin(self.angle) * self.speed, self.y)
        self.steps[alive] += 1

        off_track = alive & ~track.on_track(self.x, self.y)
        self._kill(off_track)

        still_alive = alive & ~off_track
        if still_alive.any():
            self._update_progress(still_alive)

        stalled = still_alive & (
            ((self.steps - self.last_improve) > STALL_STEPS) | (self.progress < self.best - 40)
        )
        self._kill(stalled)
        self.step_count += 1

    def _kill(self, mask: np.ndarray) -> None:
        if mask.any():
            self.death_marks.extend(zip(self.x[mask].tolist(), self.y[mask].tolist()))
            self.alive[mask] = False

    def _update_progress(self, mask: np.ndarray) -> None:
        track = self.track
        n = track.n
        candidates = (self.idx[mask, None] + _SEARCH_OFFSETS[None, :]) % n     # (m, W)
        pts = track.centerline[candidates]                                     # (m, W, 2)
        dx = pts[:, :, 0] - self.x[mask, None]
        dy = pts[:, :, 1] - self.y[mask, None]
        dist2 = dx * dx + dy * dy
        best_w = np.argmin(dist2, axis=1)
        best_idx = candidates[np.arange(len(best_w)), best_w]

        delta = best_idx - self.idx[mask]
        delta = np.where(delta > n / 2, delta - n, delta)
        delta = np.where(delta < -n / 2, delta + n, delta)

        self.idx[mask] = best_idx
        self.progress[mask] += delta

        improved = mask.copy()
        improved[mask] = self.progress[mask] > self.best[mask] + 0.4
        self.best[improved] = self.progress[improved]
        self.last_improve[improved] = self.steps[improved]
        if improved.any():
            self.gen_best = max(self.gen_best, float(self.best[improved].max()))

        lapped = mask & (self.first_lap < 0) & (self.progress >= n)
        self.first_lap[lapped] = self.steps[lapped]

    # -- generation lifecycle -----------------------------------------
    def leader_index(self) -> int:
        return int(np.argmax(np.where(self.alive, self.progress, -np.inf)))

    def is_over(self, max_steps: int) -> bool:
        if not self.alive.any():
            return True
        if self.step_count >= max_steps:
            return True
        lead = float(self.progress[self.alive].max())
        return lead >= LAPS_TO_MASTER * self.track.n

    def best_lap_seconds(self, tick_hz: float) -> float | None:
        done = self.first_lap[self.first_lap >= 0]
        if len(done) == 0:
            return None
        return float(done.min()) / tick_hz

    def lapped_fraction(self) -> float:
        return float((self.first_lap >= 0).sum()) / self.pop.size
