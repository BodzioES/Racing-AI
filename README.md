# AI Racers

**Play live: [https://racing-ai.kuncrog.com](https://racing-ai.kuncrog.com)**

Draw a track on the screen, and a population of small neural networks will
learn to drive it from scratch — generation by generation, through evolution.
No training data, no hand-written driving rules.

## What it is and what it does

AI Racers is a **neuroevolution** demo: a genetic algorithm trains the neural
network that drives each car. You open the page, draw a closed loop (or hit
the random button), and the server starts the simulation:

1. **80 cars** hit the track, each with its own random neural network.
2. The cars drive — most crash right away, a few get further.
3. When the generation ends, the cars with the **best score** (longest
   distance) are crossed and mutated, and their children drive the next
   generation.
4. After tens of generations, the cars can drive full laps.

Live you can see: all car positions, the leader's 5 sensor rays, a drawing of
its brain (neuron activations), and a progress graph over generations. You can
change simulation speed (1–20x), track width, pause, and reset evolution.

## How it works

**Split of roles:** all simulation and learning runs in **Python (NumPy)** on
the backend. The browser is a thin client: it lets you draw the track, opens
a WebSocket, and only draws the frames the server sends. There is no physics,
no neural network, and no genetic algorithm in JavaScript.

After you draw a track:

1. The frontend sends the points over a WebSocket (`/ws`, a `track` message).
2. The backend turns the sketch into a smooth, evenly spaced centerline
   (`track.py`: resampling every ~5 px + moving-average smoothing) and draws
   the track mask (Pillow). A track that is too short is rejected
   (`track_too_short`).
3. The server sends back the ready track (`track_ready`) — the frontend draws
   exactly what will be simulated.
4. The server ticks 30 times per second, runs 1–20 physics steps per tick
   (depends on the speed slider), and sends a JSON frame (`frame`): car
   positions, the leader, its sensors and weights, and score history.

Each browser tab gets its **own world** (own track and population) — two
people can use the demo at the same time without sharing state.

## The network and how it learns

A classic **feedforward 6-8-2** network (input layer → 8 hidden neurons →
2 outputs), `tanh` activation everywhere:

- **6 inputs:** 5 distance sensors (rays at −1.2, −0.6, 0, +0.6, +1.2 radians
  from the car's heading, normalized 0–1) + own speed.
- **2 outputs:** steering and throttle (both in −1…1, throttle mapped to
  min–max speed).

Learning is **not** gradients or reinforcement learning — weights only change
through genetic operators, once per generation:

- **Fitness** = longest distance along the track before dying (leaving the
  track or standing still for 90 steps).
- **Elite:** the 4 best cars pass unchanged.
- **Selection:** parents picked from the top 60% (weighted toward the best).
- **Crossover:** per-weight uniform mix with probability 0.35.
- **Mutation:** gaussian noise on ~12% of weights; when the best score stops
  improving (stagnation), mutation grows on its own.
- ~5% of each generation are fresh random networks (keeps diversity).
- A generation ends when all cars die, the step limit runs out, or the leader
  drives 2 laps.

All knobs (population size, mutation rate, car physics) live in one place:
`backend/app/config.py`.

The forward pass for the whole population is one matrix op, not a loop over
80 cars:

```python
hidden = np.tanh(np.einsum("pi,pih->ph", inputs, self.W1) + self.b1)
```

Ray-casting, physics, collisions, and progress tracking are also vectorized
for all cars at once — that is why the simulation handles 20x speed at
30 frames per second.

## Run it locally

### Docker (recommended)

```bash
git clone https://github.com/BodzioES/Racing-AI.git racing-ai
cd racing-ai
docker compose up --build
```

Page: **http://localhost:8001**, healthcheck: `http://localhost:8001/health`.
Stop: `docker compose down`.

### Without Docker (Python 3.12)

```powershell
cd backend
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Page: **http://localhost:8000**.

> Note: the project needs **Python 3.12** (same as in Docker).
> On Python 3.14 `pip install` will fail, because `numpy==2.1.3`
> has no ready packages for that version.

### Tests

```bash
cd backend
pip install pytest
pytest ../tests -v
```

## Project layout

```
frontend/
  index.html        page (PL/EN), canvas layers, HUD
  style.css         one dark theme (#121214, accent #60a5fa);
                    the canvas reads these variables live
  app.js            track drawing + WebSocket client + canvas render
                    (cars are top-down vector graphics drawn in code)

backend/
  app/
    config.py         all tunable constants
    track.py          sketch -> smooth centerline -> track mask
    neuroevolution.py 6-8-2 population + selection/crossover/mutation
    simulation.py     physics, sensors, progress — vectorized
    world.py          one track + generations; one World per connection
    main.py           FastAPI: frontend, /health, /ws loop
  requirements.txt
  Dockerfile          python:3.12-slim + uvicorn on port 8000

tests/
  test_track_and_sim.py  6 tests: track geometry, forward pass, elite,
                         off-track death, full World -> frame run

docker-compose.yml    app on 127.0.0.1:8001 (behind Nginx in production)
```

In production the project runs at
**[racing-ai.kuncrog.com](https://racing-ai.kuncrog.com)**
(Nginx + HTTPS + Cloudflare, deployed from GitHub Actions on every push
to `main`).
