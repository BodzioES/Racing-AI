"""One World = one viewer's track + the whole generation-after-generation
evolution running against it. A fresh World is created per WebSocket
connection so two people drawing tracks at once never share state.
"""
from __future__ import annotations

import numpy as np

from . import track as track_mod
from .config import MAX_SPEED, POP, TICK_HZ
from .neuroevolution import Population
from .simulation import Generation, Track


class World:
    def __init__(self):
        self.track: Track | None = None
        self.gen_number = 1
        self.generation: Generation | None = None
        self.all_time_best = 0.0
        self.stagnation = 0
        self.history: list[dict] = []
        self.max_steps = 3000
        self.rng = np.random.default_rng()

    def set_track(self, raw_points, width: float, w: int, h: int) -> None:
        centerline = track_mod.build_centerline(raw_points)
        mask = track_mod.build_mask(centerline, width, w, h)
        angle = track_mod.start_heading(centerline)
        self.track = Track(centerline, mask, width, angle)
        self.max_steps = max(1800, round(self.track.n * 5 * 2 / 2.2))
        self.reset_evolution()

    def reset_evolution(self) -> None:
        self.gen_number = 1
        self.all_time_best = 0.0
        self.stagnation = 0
        self.history = []
        population = Population(POP, rng=self.rng)
        self.generation = Generation(self.track, population)

    def tick(self) -> bool:
        """Advance one simulation step. Returns True if a new generation started."""
        gen = self.generation
        gen.step()
        if gen.is_over(self.max_steps):
            self._advance_generation()
            return True
        return False

    def _advance_generation(self) -> None:
        gen = self.generation
        fitness = gen.best
        top = float(fitness.max())
        avg = float(fitness.mean())
        n = self.track.n
        self.history.append({"best": top / n, "avg": avg / n})
        if len(self.history) > 150:
            self.history.pop(0)

        if top > self.all_time_best + 1:
            self.all_time_best = top
            self.stagnation = 0
        elif self.all_time_best < 2 * n - 8:
            self.stagnation += 1

        next_pop = gen.pop.next_generation(fitness, mutation_boost=self.stagnation)
        self.gen_number += 1
        self.generation = Generation(self.track, next_pop)

    def frame(self) -> dict:
        gen = self.generation
        leader = gen.leader_index()
        n = self.track.n
        inp = np.append(gen.rays[leader], gen.speed[leader] / MAX_SPEED)
        w1, b1 = gen.pop.W1[leader], gen.pop.b1[leader]
        w2, b2 = gen.pop.W2[leader], gen.pop.b2[leader]
        hidden = np.tanh(inp @ w1 + b1)
        output = np.tanh(hidden @ w2 + b2)
        return {
            "type": "frame",
            "gen": self.gen_number,
            "alive": int(gen.alive.sum()),
            "pop": self.generation.pop.size,
            "bestLaps": round(max(self.all_time_best, gen.gen_best) / n, 3),
            "bestLapSec": gen.best_lap_seconds(TICK_HZ),
            "lapFraction": gen.lapped_fraction(),
            "leader": leader,
            "leaderRays": gen.rays[leader].tolist(),
            "leaderInputSpeed": float(gen.speed[leader]),
            "leaderBrain": gen.pop.brain_of(leader),
            "leaderInput": inp.tolist(),
            "leaderHidden": hidden.tolist(),
            "leaderOutput": output.tolist(),
            "leaderPos": [round(float(gen.x[leader]), 1), round(float(gen.y[leader]), 1), round(float(gen.angle[leader]), 3)],
            "cars": [
                [round(float(x), 1), round(float(y), 1), round(float(a), 3)]
                for x, y, a, alive in zip(gen.x, gen.y, gen.angle, gen.alive)
                if alive
            ],
            "deaths": gen.death_marks[-40:],
            "history": self.history[-1:],
        }
