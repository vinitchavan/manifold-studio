# FastAPI backend

Run from the **repository root** (Python 3.10+):

```bash
python -m pip install -e ".[viz,train]"
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 for the UI; `/docs` contains the API schema.
For text input, additionally install `python -m pip install -e ".[text]"` and
restart. First text use downloads `sentence-transformers/all-MiniLM-L6-v2`.
The synthetic and numeric modes do not download models. Training needs PyTorch;
PCA mapping works without it. The catalog tells the UI which extras are installed.

## Routes

- `GET /api/health`: dependency readiness.
- `GET /api/catalog`: geometry, loss options and limits.
- `POST /api/run`: validated config -> Plotly figure JSON, metrics, training history.
- `POST /api/export`: replay config -> ZIP with NPZ embeddings, PCA/trained mapper,
  CSV coordinates, metrics and a standalone offline HTML graph.

Example JSON:

```json
{"source":"demo","mode":"train","geometry":["sphere","torus"],"metric":"intrinsic","loss":"smtl","steps":30,"seed":7}
```

For `source="vectors"`, supply `vectors` as a numeric matrix and optional `texts`
(one per row). For text mode supply `texts`. For user-data triplet/SMTL training,
provide explicit `triplets` (zero-based row indices). The demo alone creates its
own synthetic class triplets. Spectral objectives use `query_index` (a selected
input row) as the query. `target` is the tangent-arrow target; null hides arrows.
Möbius requires `metric="ambient"`. Euclidean/ambient triplet losses require
ambient distance. Geodesic triplet requires intrinsic distance.

This is a **local single-user prototype**, not an authenticated hosted service.
Keep the default loopback binding. Runs are synchronous and serialized with a
busy response (429). Limits: 256 rows, 2,048 source dimensions, three factors and
100 training steps. No training cancellation or background job persistence yet.
Requests process in a worker thread so health/static routes can remain available.

Files exist only in a temporary directory while a request runs; exports are
returned immediately and temporary files removed. The server retains no run
history. Download replays the last successful UI configuration; it is not a
stored snapshot and exact model reproducibility across library versions is not
guaranteed. Input payloads are not written to a shared output folder. No wildcard
CORS is enabled. Plotly JavaScript is served locally from the installed package.

The response reports neighborhood preservation and training objectives, **not
semantic accuracy**. No held-out task evaluation is implemented in this UI.
See `docs/LOSSES.md` for objective assumptions and the notebook for a held-out
example. Text-mode model downloads/real encoding have not been tested end to end;
API tests replace the encoder with a fixture to avoid network/model downloads.

Tests (from root):

```bash
python -m unittest discover -s backend/tests -v
```
