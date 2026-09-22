"""Tunable constants for the simulation and the neuroevolution loop."""
import numpy as np

POP = 80          # population size (cars per generation)
NIN = 6           # inputs: 5 ray sensors + own speed
NH = 8            # hidden units
NOUT = 2          # outputs: steer, throttle

MIN_SPEED = 1.3
MAX_SPEED = 5.2
TURN_RATE = 0.105
RAY_ANGLES = np.array([-1.2, -0.6, 0.0, 0.6, 1.2], dtype=np.float64)
RAY_STEP = 3.0          # px per ray-marching step
STALL_STEPS = 90        # steps without progress before a car is killed
LAPS_TO_MASTER = 2       # laps a leader must complete to end a generation
CENTERLINE_SPACING = 5   # px between resampled centerline points
SEARCH_WINDOW = range(-6, 15)  # offsets checked when locating nearest centerline point

INIT_WEIGHT_SCALE = 0.4  # small initial weights so learning is visible, not instant
MUTATION_RATE = 0.12
MUTATION_SIGMA = 0.25
ELITE_KEEP = 4
FRESH_RANDOM = max(1, round(POP * 0.05))
CROSSOVER_PROB = 0.35

TICK_HZ = 30             # server ticks per second (frames sent to the client)
