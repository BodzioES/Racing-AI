"""A population of tiny (6-8-2) feedforward networks, evolved with a
genetic algorithm. No gradients, no training data -- every car's brain
is a set of weights that only ever changes through selection, crossover
and mutation.

Everything here is vectorized over the whole population at once with
NumPy: the forward pass for all 80 cars is a single batched einsum, not
an 80-iteration Python loop.
"""
from __future__ import annotations

import numpy as np

from .config import (
    CROSSOVER_PROB,
    ELITE_KEEP,
    FRESH_RANDOM,
    INIT_WEIGHT_SCALE,
    MUTATION_RATE,
    MUTATION_SIGMA,
    NH,
    NIN,
    NOUT,
    POP,
)


class Population:
    """Weights for `size` networks, stored as batched arrays (not a list of objects)."""

    def __init__(self, size: int = POP, rng: np.random.Generator | None = None):
        self.size = size
        self.rng = rng or np.random.default_rng()
        self.W1, self.b1, self.W2, self.b2 = self._random_weights(size)

    def _random_weights(self, n: int):
        scale = INIT_WEIGHT_SCALE
        W1 = self.rng.uniform(-scale, scale, (n, NIN, NH))
        b1 = self.rng.uniform(-scale, scale, (n, NH))
        W2 = self.rng.uniform(-scale, scale, (n, NH, NOUT))
        b2 = self.rng.uniform(-scale, scale, (n, NOUT))
        return W1, b1, W2, b2

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """inputs: (size, NIN) -> (hidden (size, NH), output (size, NOUT)), batched over the population."""
        hidden = np.tanh(np.einsum("pi,pih->ph", inputs, self.W1) + self.b1)
        output = np.tanh(np.einsum("ph,pho->po", hidden, self.W2) + self.b2)
        return hidden, output

    def brain_of(self, index: int) -> dict:
        """Single car's weights, JSON-friendly, for the live "leader's brain" view."""
        return {
            "w1": self.W1[index].tolist(),
            "b1": self.b1[index].tolist(),
            "w2": self.W2[index].tolist(),
            "b2": self.b2[index].tolist(),
        }

    def next_generation(self, fitness: np.ndarray, mutation_boost: float = 0.0) -> "Population":
        """Rank by fitness, keep an elite, breed the rest, mutate, return a fresh Population."""
        order = np.argsort(-fitness)
        n = self.size
        child = Population.__new__(Population)
        child.size = n
        child.rng = self.rng
        W1 = np.empty_like(self.W1)
        b1 = np.empty_like(self.b1)
        W2 = np.empty_like(self.W2)
        b2 = np.empty_like(self.b2)

        for i in range(ELITE_KEEP):
            src = order[i]
            W1[i], b1[i], W2[i], b2[i] = self.W1[src], self.b1[src], self.W2[src], self.b2[src]

        fresh_start = n - FRESH_RANDOM
        rW1, rb1, rW2, rb2 = self._random_weights(FRESH_RANDOM)
        W1[fresh_start:], b1[fresh_start:] = rW1, rb1
        W2[fresh_start:], b2[fresh_start:] = rW2, rb2

        pool = order[: max(2, round(n * 0.6))]
        for i in range(ELITE_KEEP, fresh_start):
            a = pool[int(self.rng.power(1 / 2.2) * len(pool))]
            if self.rng.random() < CROSSOVER_PROB:
                b = pool[int(self.rng.power(1 / 2.2) * len(pool))]
                mask1 = self.rng.random((NIN, NH)) < 0.5
                W1[i] = np.where(mask1, self.W1[a], self.W1[b])
                b1[i] = np.where(self.rng.random(NH) < 0.5, self.b1[a], self.b1[b])
                mask2 = self.rng.random((NH, NOUT)) < 0.5
                W2[i] = np.where(mask2, self.W2[a], self.W2[b])
                b2[i] = np.where(self.rng.random(NOUT) < 0.5, self.b2[a], self.b2[b])
            else:
                W1[i], b1[i], W2[i], b2[i] = self.W1[a], self.b1[a], self.W2[a], self.b2[a]

        rate = min(0.85, MUTATION_RATE + mutation_boost * 0.05)
        sigma = min(1.2, MUTATION_SIGMA + mutation_boost * 0.02)
        for arr in (W1, b1, W2, b2):
            arr_slice = arr[ELITE_KEEP:]
            hits = self.rng.random(arr_slice.shape) < rate
            arr_slice += hits * self.rng.normal(0.0, sigma, arr_slice.shape)

        child.W1, child.b1, child.W2, child.b2 = W1, b1, W2, b2
        return child
