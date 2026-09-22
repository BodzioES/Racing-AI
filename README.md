# AI Racers

Draw a racetrack anywhere on the screen. A population of small neural
networks learns to drive it from scratch, generation by generation, with a
genetic algorithm (neuroevolution) -- no training data, no hand-written
driving rules.

**Split on purpose:** all simulation and learning run in **Python (NumPy)**
on the backend. The browser is a thin client: it lets you draw the track,
opens a WebSocket, and only draws whatever frames the server sends. Nothing
about the cars, the sensors, the neural nets, or the genetic algorithm runs
in JavaScript.

## Architecture

```
frontend/            static HTML/CSS/JS, served by the backend
  index.html
  style.css
  app.js             drawing UI + WebSocket client + canvas rendering only

backend/
  app/
    config.py        all tunable constants in one place
    track.py         hand-drawn points -> smooth closed centerline -> pixel mask (Pillow)
    neuroevolution.py  population of 6-8-2 nets as batched NumPy arrays
                        + selection / crossover / mutation
    simulation.py    car physics, ray-sensors, progress tracking -- vectorized
                        over the whole population (no per-car Python loop)
    world.py         ties one track + the running generations together;
                        one World per WebSocket connection
    main.py          FastAPI app: serves the frontend, runs the /ws loop
  requirements.txt
  Dockerfile

tests/
  test_track_and_sim.py   pytest: track geometry, forward pass, genetic
                            operators, generation-end conditions

docker-compose.yml   tylko aplikacja na 127.0.0.1:8000 (bez Caddy)
nginx/               przykladowy vhost dla wlasnego Nginx na VPS
  racing-ai.kuncrog.com.conf
.github/workflows/
  deploy.yml         deploy na VPS: ssh + git pull + compose up
```

### Protocol (over one WebSocket per tab, at `/ws`)

Client -> server:
- `{"type":"track","points":[[x,y],...],"width":70,"w":1280,"h":720}` -- sent once the drawn loop closes
- `{"type":"speed","value":12}` -- simulation speed multiplier, 1-20
- `{"type":"pause","value":true}`
- `{"type":"reset"}` -- new random population on the same track

Server -> client:
- `{"type":"track_ready","centerline":[...],"width":70,"startAngle":1.02}` -- the actual track the simulation will use (already resampled/smoothed), so the frontend draws exactly what's being simulated
- `{"type":"track_too_short"}`
- `{"type":"frame", gen, alive, pop, bestLaps, bestLapSec, lapFraction, leaderPos, leaderRays, leaderBrain, leaderInput, leaderHidden, leaderOutput, cars, deaths, history}` -- sent ~30x/second

## Why this is vectorized, not just "translated to Python"

The forward pass for the whole population is one batched `einsum`, not an
80-iteration loop:

```python
hidden = np.tanh(np.einsum("pi,pih->ph", inputs, self.W1) + self.b1)
```

Ray-casting, physics, collision checks and nearest-centerline lookups are
all done as whole-array NumPy operations across every car at once. This is
also *why* it can run at up to 20x simulation speed and still keep up with
a 30 Hz WebSocket stream.

## Running it

### With Docker (recommended, matches production)

```bash
git clone https://github.com/BodzioES/Racing-AI.git racing-ai
cd racing-ai
docker compose up --build -d
```

This starts one container: the app on `127.0.0.1:8001` (not exposed
publicly). Public traffic goes through **your own Nginx** on the same VPS:

`racing-ai.kuncrog.com` -> Nginx (80/443, HTTPS) -> `127.0.0.1:8001`.

(port 8001, bo 8000 zajmuje juz portfolio-web-1 na tym VPS).

Nginx vhost: [`nginx/racing-ai.kuncrog.com.conf`](nginx/racing-ai.kuncrog.com.conf).
Copy it to `/etc/nginx/sites-available/`, symlink to `sites-enabled`,
then `certbot --nginx -d racing-ai.kuncrog.com`.

DNS: `racing-ai.kuncrog.com` A record -> your VPS IP (already set in OVH).

**WebSocket is mandatory** -- `/ws` needs `Upgrade`/`Connection` headers,
otherwise the page sits on "Connecting to server...". The bundled Nginx
config already has that for `/` (so `/ws` works too). Minimal snippet:

```nginx
location / {
    proxy_pass http://127.0.0.1:8001;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

### Deploy na VPS (auto przez GitHub Actions)

Workflow [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)
robi na kazdy push do `main`: SSH -> `git pull` w `~/racing-ai` ->
`docker compose up --build -d`.

Ustaw w GitHub: repo -> Settings -> Secrets and variables -> Actions:

- Secrets: `VPS_HOST`, `VPS_USERNAME`, `VPS_SSH_KEY`
- Variables (opcjonalnie): `VPS_PORT` (domyslnie `22`), `VPS_PROJECT_DIR` (domyslnie `~/racing-ai`)

### Without Docker (local dev)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000`.

### Tests

```bash
cd backend
pip install pytest
pytest ../tests -v
```

## Honest status -- read this before you demo it

I (the AI that wrote this) built and unit-tested the **logic** thoroughly:

- `track.py`, `neuroevolution.py`, `simulation.py`, `world.py` were run
  directly in Python (NumPy is available in the sandbox I built this in).
  All 6 tests in `tests/test_track_and_sim.py` pass.
- I ran the full generation-by-generation evolution against several
  synthetic tracks (an ellipse and multiple randomly-shaped loops) and
  confirmed cars go from ~0.1 laps in generation 1 to mastering the track
  (2 full laps, 30%+ of the population lapping) within roughly 10-36
  generations, depending on track shape. That's slower than the original
  all-JavaScript prototype (which used a slightly different mutation
  schedule); if you want it to converge faster, the easiest knobs are in
  `config.py` (`MUTATION_RATE`, `MUTATION_SIGMA`, `INIT_WEIGHT_SCALE`).
- Throughput was in the thousands of simulation steps per second on a
  single CPU core, comfortably enough for 20x speed at a 30 Hz frame rate.

**What I could not test here:** my sandbox has no internet access and
doesn't have `fastapi`/`uvicorn`/`websockets` installed, and no browser. So
`main.py` (the actual WebSocket server) and `app.js` (the actual browser
client) are written carefully and match the tested logic's data shapes and
message protocol exactly -- but I have never run them end-to-end together,
and I have never opened the page in a real browser. Realistic things that
could still be wrong: a message-shape mismatch I typo'd on one side, a
CSS/canvas sizing issue on your monitor or phone, or a proxy/WebSocket
config issue on your specific server setup.

**Before you show this to anyone:** run `docker compose up --build`, open
the page yourself, draw a few tracks (including a small one and a very
loopy one), leave it running for a few minutes, and try it on your phone.
If anything breaks, tell me exactly what happened (an error in the browser
console is the most useful thing you can paste back) and I'll fix it.

## Talking about this in an interview

- It's a genetic algorithm (neuroevolution), not reinforcement learning or
  supervised learning -- there's no reward signal being backpropagated and
  no labeled dataset. Say that plainly; it's still a legitimate, classic
  approach and it's honest.
- Be ready to explain: why a 6-8-2 network, what the 5 ray-sensors are and
  why they're relative to the car's heading, what "fitness" is here
  (distance traveled along the track before dying or stalling), how
  selection/crossover/mutation produce the next generation, and why the
  simulation is vectorized with NumPy instead of a Python loop per car.
- Reasonable follow-up question: "why not reinforcement learning?"
  Neuroevolution needs no gradients, is simple to implement and to reason
  about, and parallelizes trivially across a population -- a fair trade
  against typically needing more total simulated steps than RL to reach
  the same result.
