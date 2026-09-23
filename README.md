# AI Racers

**Zagraj na żywo: [https://racing-ai.kuncrog.com](https://racing-ai.kuncrog.com)**

Narysuj tor na ekranie, a populacja małych sieci neuronowych nauczy się po nim
jeździć od zera — pokolenie za pokoleniem, przez ewolucję. Bez danych
treningowych, bez wpisanych reguł jazdy.

## Czym jest ten projekt i co robi

AI Racers to demo **neuroewolucji**: algorytm genetyczny trenuje sieć neuronową
sterującą autkiem. Otwierasz stronę, rysujesz zamkniętą pętlę (lub losujesz
przyciskem), a serwer zaczyna symulację:

1. Na tor wyjeżdża **80 aut**, każde z własną, losową siecią neuronową.
2. Auta jadą — większość od razu wypada z toru, kilka dojedzie dalej.
3. Gdy pokolenie się kończy, auta z **najlepszym wynikiem** (najdłuższy dystans)
   są krzyżowane i mutowane, a ich potomstwo jedzie w następnym pokoleniu.
4. Po kilkunastu–kilkudziesięciu pokoleniach auta potrafią przejechać pełne
   okrążenia.

Na żywo widać: pozycje wszystkich aut, 5 promieni-sensorów lidera, wizualizację
jego mózgu (aktywacje neuronów) oraz wykres postępu w kolejnych pokoleniach.
Można zmieniać prędkość symulacji (1–20×), szerokość toru, pauzować i resetować
ewolucję.

## Jak to działa

**Podział ról:** cała symulacja i uczenie liczą się w **Pythonie (NumPy)** na
backendzie. Przeglądarka to cienki klient: pozwala narysować tor, otwiera
WebSocket i tylko rysuje klatki, które przysyła serwer. W JavaScript nie ma
ani fizyki, ani sieci neuronowych, ani algorytmu genetycznego.

Przebieg po narysowaniu toru:

1. Frontend wysyła punkty przez WebSocket (`/ws`, wiadomość `track`).
2. Backend zamienia bazgroł w gładką, równo rozłożoną linię środkową
   (`track.py`: resampling co ~5 px + wygładzanie średnią ruchomą) i rasteryzuje
   maskę toru (Pillow). Za krótki tor jest odrzucany (`track_too_short`).
3. Serwer odsyła gotowy tor (`track_ready`) — frontend rysuje dokładnie to,
   co będzie symulowane.
4. Serwer tyka 30 razy na sekundę, na każdy tick robi 1–20 kroków fizyki
   (zależnie od suwaka prędkości) i wysyła klatkę JSON (`frame`): pozycje aut,
   lidera, jego sensory i wagi, historię wyników.

Każda karta przeglądarki dostaje **osobny świat** (osobny tor i populację) —
dwie osoby mogą używać demo naraz bez dzielenia stanu.

## Jaka to sieć i jak się uczy

Klasyczny **feedforward 6-8-2** (warstwa wejściowa → 8 neuronów ukrytych →
2 wyjścia), aktywacja `tanh` wszędzie:

- **6 wejść:** 5 czujników odległości (promienie pod kątami
  −1.2, −0.6, 0, +0.6, +1.2 radiana względem kierunku auta, znormalizowane
  0–1) + własna prędkość.
- **2 wyjścia:** skręt i gaz (oba w zakresie −1…1, gaz mapowany na prędkość
  minimalną–maksymalną).

Uczenie to **nie** gradient ani reinforcement learning — wagi zmieniają się
tylko przez operatory genetyczne, raz na pokolenie:

- **Fitness** = najdalszy dystans wzdłuż toru przed śmiercią (wypadnięcie
  z toru albo stanie w miejscu przez 90 kroków).
- **Elita:** 4 najlepsze auta przechodzą bez zmian.
- **Selekcja:** rodzice losowani z top 60% (z wagą na najlepszych).
- **Krzyżowanie:** uniform per-waga z prawdopodobieństwem 0.35.
- **Mutacja:** szum gaussowski na ~12% wag; przy stagnacji (brak poprawy
  najlepszego wyniku) mutacja rośnie automatycznie.
- ~5% każdego pokolenia to świeże, losowe sieci (utrzymują różnorodność).
- Pokolenie kończy się, gdy wszystkie auta zginą, skończy się limit kroków
  albo lider przejedzie 2 okrążenia.

Wszystkie stałe (rozmiar populacji, tempo mutacji, fizyka auta) są w jednym
miejscu: `backend/app/config.py`.

Forward dla całej populacji to jedna operacja macierzowa, nie pętla po
80 autach:

```python
hidden = np.tanh(np.einsum("pi,pih->ph", inputs, self.W1) + self.b1)
```

Tak samo ray-casting, fizyka, kolizje i pomiar postępu liczą się wektorowo
dla wszystkich aut naraz — dlatego symulacja wyrabia 20× prędkości przy
30 klatkach na sekundę.

## Jak odpalić lokalnie

### Docker (polecane)

```bash
git clone https://github.com/BodzioES/Racing-AI.git racing-ai
cd racing-ai
docker compose up --build
```

Strona: **http://localhost:8001**, healthcheck: `http://localhost:8001/health`.
Zatrzymanie: `docker compose down`.

### Bez Dockera (Python 3.12)

```powershell
cd backend
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Strona: **http://localhost:8000**.

> Uwaga: projekt wymaga **Pythona 3.12** (taki sam jest w Dockerze).
> Na Pythonie 3.14 `pip install` się nie powiedzie, bo `numpy==2.1.3`
> nie ma gotowych pakietów dla tej wersji.

### Testy

```bash
cd backend
pip install pytest
pytest ../tests -v
```

## Struktura projektu

```
frontend/
  index.html        strona (PL/EN), warstwy canvas, HUD
  style.css         jeden ciemny motyw (#121214, akcent #60a5fa);
                    canvas czyta te zmienne na żywo
  app.js            rysowanie toru + klient WebSocket + render canvas
                    (auta to wektorowe grafiki top-down rysowane kodem)

backend/
  app/
    config.py         wszystkie przestrajalne stałe
    track.py          bazgroł -> gładka linia środkowa -> maska toru
    neuroevolution.py populacja sieci 6-8-2 + selekcja/krzyżowanie/mutacja
    simulation.py     fizyka, sensory, postęp — wektorowo dla całej populacji
    world.py          jeden tor + kolejne pokolenia; jeden World na połączenie
    main.py           FastAPI: frontend, /health, pętla /ws
  requirements.txt
  Dockerfile          python:3.12-slim + uvicorn na porcie 8000

tests/
  test_track_and_sim.py  6 testów: geometria toru, forward, elita,
                         śmierć poza torem, pełny przebieg World -> frame

docker-compose.yml    aplikacja na 127.0.0.1:8001 (na produkcji za Nginx)
```

Produkcyjnie projekt działa pod adresem
**[racing-ai.kuncrog.com](https://racing-ai.kuncrog.com)**
(Nginx + HTTPS + Cloudflare, deploy z GitHub Actions przy każdym pushu na `main`).
