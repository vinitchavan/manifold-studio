"""Local single-user workbench. Run: uvicorn backend.main:app --host 127.0.0.1."""
import importlib.util
import io
import json
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from plotly.offline import get_plotlyjs
from manifold_studio import ManifoldProjector, neighbor_preservation
from manifold_studio.cli import parse_geometry

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Manifold Studio", version="0.3.0")
# Serialize bounded compute runs, including PyTorch's process-global RNG.
RUN_LOCK = threading.Lock()


class RunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    mode: Literal['map', 'train'] = 'map'
    source: Literal['demo', 'vectors', 'text'] = 'demo'
    geometry: list[Literal['sphere', 'torus', 'cylinder', 'mobius', 'plane', 'circle']] = Field(default=['sphere'], min_length=1, max_length=3)
    metric: Literal['intrinsic', 'ambient'] = 'intrinsic'
    vectors: list[list[float]] | None = Field(default=None, max_length=256)
    texts: list[str] | None = Field(default=None, max_length=256)
    labels: list[str] | None = Field(default=None, max_length=256)
    loss: str = Field(default='geodesic_triplet', max_length=60)
    steps: int = Field(default=30, ge=1, le=100)
    margin: float = Field(default=0.3, ge=0, le=10)
    alpha: float = Field(default=1, ge=0, le=10)
    beta: float = Field(default=0.5, ge=0, le=10)
    gamma: float = Field(default=0.3, ge=0, le=10)
    seed: int = Field(default=7, ge=0, le=2147483647)
    target: int | None = Field(default=0, ge=0, le=255)
    show_labels: bool = False
    triplets: list[list[int]] | None = Field(default=None, max_length=1024)
    query_index: int = Field(default=0, ge=0, le=255)


def inputs(req):
    labels = req.labels
    if req.source == 'demo':
        rng = np.random.default_rng(req.seed)
        groups = np.repeat(np.arange(3), 16)
        x = rng.normal(size=(3, 16))[groups] + .4*rng.normal(size=(48, 16))
        texts = [f'Group {g+1} · point {i+1}' for i, g in enumerate(groups)]
        labels = groups.astype(str).tolist()
    elif req.source == 'text':
        if not req.texts or any(not t.strip() or len(t) > 4000 for t in req.texts):
            raise ValueError('Enter one nonempty sentence per line (maximum 4,000 characters each).')
        from manifold_studio.encoders import encode_sentences
        x = encode_sentences(req.texts)
        texts = req.texts
    else:
        if req.vectors is None or any(len(row) > 2048 for row in req.vectors):
            raise ValueError('Supply a numeric matrix with at most 2,048 columns.')
        x = np.asarray(req.vectors, dtype=float)
        texts = req.texts
    if x.ndim != 2 or not 4 <= len(x) <= 256 or not 1 <= x.shape[1] <= 2048 or not np.isfinite(x).all():
        raise ValueError('Use a finite matrix with 4–256 rows and 1–2,048 columns.')
    if texts is not None and (len(texts) != len(x) or any(len(t) > 4000 for t in texts)):
        raise ValueError('Text labels must match the input row count and be at most 4,000 characters.')
    if labels is not None and len(labels) != len(x):
        raise ValueError('Class labels must match input rows.')
    if req.target is not None and req.target >= len(x):
        raise ValueError('Tangent target is outside the input rows.')
    return x, texts, labels


def compute(req, directory):
    geometry = parse_geometry('*'.join(req.geometry))
    if req.metric == 'intrinsic' and not geometry.supports_intrinsic:
        raise ValueError('Möbius requires ambient distance, including in a product.')
    x, texts, labels = inputs(req)
    history = []
    if req.mode == 'map':
        model = ManifoldProjector(geometry).fit(x)
        embedding = model.transform(x, texts=texts)
        model.save(directory/'projector.npz')
    else:
        import torch
        from manifold_studio.training import TrainableProjector
        from manifold_studio.losses import make_loss, fixed_spectral_basis
        objective = make_loss(req.loss, geometry, metric=req.metric, margin=req.margin,
                              alpha=req.alpha, beta=req.beta, gamma=req.gamma)
        triplets = req.triplets
        if triplets is None and req.source == 'demo':
            triplets = [[i, (i//16)*16+(i+1)%16, (i+16)%48] for i in range(48)]
        if (req.loss.endswith('triplet') or req.loss == 'smtl') and triplets is None:
            raise ValueError('Provide triplet indices for uploaded data; no labels are invented.')
        spectral = req.loss in ('query_energy_separation', 'energy_concentration') or (req.loss == 'smtl' and (req.alpha > 0 or req.beta > 0))
        if spectral and req.query_index >= len(x):
            raise ValueError('Query index is outside the input rows.')
        X = torch.tensor(x, dtype=torch.float32)
        basis = fixed_spectral_basis(X) if spectral else None
        # This endpoint is serialized; preserve the caller's RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(req.seed)
            model = TrainableProjector(x.shape[1], geometry).fit_normalization(X)
            optimizer = torch.optim.Adam(model.parameters(), lr=.005)
            for step in range(req.steps+1):
                result = objective(model(X), triplets=triplets, source=X, basis=basis,
                    query_parameters=model(X[req.query_index:req.query_index+1]) if spectral else None)
                history.append({'step': step, **result.detached()})
                if step < req.steps:
                    optimizer.zero_grad()
                    result.total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                    optimizer.step()
            model.eval().fit_display(X)
            embedding = model.transform(X, texts=texts)
            model.save(directory/'trained_projector.npz')
    metrics = neighbor_preservation(x, embedding, k=min(5, len(x)-1), metric=req.metric)
    report = dict(mode=req.mode, source=req.source, geometry=geometry.spec(), metric=req.metric,
                  rows=len(x), source_dim=x.shape[1], intrinsic_dim=geometry.intrinsic_dim,
                  ambient_dim=geometry.ambient_dim, metrics=metrics, history=history,
                  interpretation='Neighborhood overlap and training objective, not held-out semantic accuracy.',
                  configuration=req.model_dump(exclude={'vectors', 'texts', 'labels', 'triplets'}))
    embedding.save(directory/'embeddings.npz')
    embedding.export_csv(directory/'coordinates.csv')
    (directory/'metrics.json').write_text(json.dumps(report, indent=2))
    fig = embedding.plot_3d(labels=labels, target_index=req.target, show_labels=req.show_labels)
    fig.update_layout(height=600, paper_bgcolor='#101a2b', margin=dict(l=10,r=10,t=130,b=40))
    return fig, report


@app.get('/api/health')
def health():
    return {'status': 'ok', 'training': importlib.util.find_spec('torch') is not None,
            'text': importlib.util.find_spec('sentence_transformers') is not None}


@app.get('/api/catalog')
def catalog():
    losses = ['euclidean_triplet', 'geodesic_triplet', 'ambient_triplet', 'distance_alignment',
              'query_energy_separation', 'energy_concentration', 'smtl']
    return {'geometries': ['sphere','torus','cylinder','mobius','plane','circle'], 'losses': losses,
            'limits': {'rows':256,'columns':2048,'steps':100,'factors':3}, **health()}


def run(req, download=False):
    if not RUN_LOCK.acquire(blocking=False):
        raise HTTPException(429, 'Another run is active. Try again when it finishes.')
    try:
        with tempfile.TemporaryDirectory(prefix='manifold-studio-') as tmp:
            directory = Path(tmp)
            fig, report = compute(req, directory)
            if download:
                fig.write_html(directory/'explorer.html', include_plotlyjs=True)
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
                    for path in sorted(directory.iterdir()): archive.write(path, path.name)
                return Response(buffer.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition':'attachment; filename="manifold-studio-run.zip"'})
            return {'plot': json.loads(fig.to_json()), 'report': report}
    except ImportError as exc:
        raise HTTPException(503, 'Missing optional dependency. Install .[train] for training or .[text] for sentences.') from exc
    except (ValueError, NotImplementedError, TypeError, IndexError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        RUN_LOCK.release()


@app.post('/api/run')
def execute(req: RunRequest):
    return run(req)


@app.post('/api/export')
def export(req: RunRequest):
    """Replays the configuration, then returns a ZIP; no server-side run retention."""
    return run(req, download=True)


@app.get('/vendor/plotly.js', include_in_schema=False)
def plotly_js():
    return Response(get_plotlyjs(), media_type='application/javascript', headers={'Cache-Control':'public, max-age=86400'})


@app.get('/', include_in_schema=False)
def index():
    return FileResponse(ROOT/'frontend/index.html')


app.mount('/static', StaticFiles(directory=ROOT/'frontend'), name='frontend')
