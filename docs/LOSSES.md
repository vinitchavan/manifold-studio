# Selectable research losses (v0.2)

Install `python -m pip install -e '.[train,viz]'` from this repository.
The NumPy/PCA workflow remains usable without PyTorch. Import training explicitly:

```python
from manifold_studio import Sphere, Torus
from manifold_studio.losses import available_losses, make_loss, fixed_spectral_basis
from manifold_studio.training import TrainableProjector

print(available_losses())
geometry = Sphere() * Torus()
objective = make_loss('smtl', geometry, margin=0.3, alpha=1, beta=0.5, gamma=0.3)
```

## Choose an objective

| Name | Required supervision | Role and provenance |
| --- | --- | --- |
| `euclidean_triplet` | Explicit `(anchor, positive, negative)` indices | Squared ambient distances; matches the formula in the 2025 repository |
| `geodesic_triplet` | Explicit triplets | Intrinsic distances on supported geometries; start here for metric learning |
| `ambient_triplet` | Explicit triplets | Unsquared chord distances; explicit choice for Möbius |
| `distance_alignment` | Frozen source embeddings | Preserve normalized pairwise geometry; adapted from SFM's “curvature alignment” |
| `query_energy_separation` | Projected query and fixed spectral basis | Penalize energy in the highest-frequency third of modes; experimental regularizer |
| `energy_concentration` | Projected query and fixed spectral basis | Minimize normalized spectral entropy; experimental regularizer |
| `smtl` | Triplets, source embeddings, projected query, fixed basis | Weighted combination of the four SFM components |

There is no universally best objective established here. Use triplet loss as a
baseline and SMTL as an experimental comparison. Spectral-only minimization can
collapse distinctions; low entropy is not proof of useful retrieval. Select loss,
geometry and weights on validation data, then report held-out results. Historical
notebook scores are not results of this implementation. Focal/classification losses
and history-only anchor/area/curvature proposals are not silently presented as
verified custom embedding losses.

## Contract and formulas

`objective(parameters, triplets=..., source=..., query_parameters=..., basis=...)`
returns `LossResult(total, components)`. Call `total.backward()` for training and
`detached()` for logging. Inputs are float32/float64 PyTorch tensors on one device.
`parameters` are N × intrinsic dimension; queries are 1 × intrinsic dimension.
Do not pass ambient vectors or 3D PCA display positions as parameters.

- Triplet: `mean(relu(d(a,p) - d(a,n) + margin))`. Only `euclidean_triplet`
  squares both distances. Margin defaults to **0.3** in the chosen distance units;
  set **1.0** for the original `train.py` default or **0.5** for its notebooks.
  No automatic distance/margin scaling is applied.
- Alignment: over all unordered off-diagonal pairs, mean squared difference of
  `d / max(d)` and source cosine distance divided by its maximum. Source vectors
  are detached. This is distance preservation, **not a curvature estimator**.
- Spectral field: `g = softmax(-d(document,query)/temperature)` and
  `F = g[:,None] * ambient_coordinates`. For the fixed basis `U`, mode energies
  are `E[k] = sum((U.T @ F)[k]**2)` and `P = E / sum(E)`.
- Separation: sum of `P` in the final `max(1, floor(N * noise_fraction))` modes,
  default `noise_fraction=1/3`. “High frequency = noise” is a hypothesis, not a
  label oracle; modes are ordered by ascending Laplacian eigenvalue.
- Concentration: `-sum(P*log(P))/log(N)` using all N modes, including zero-energy
  modes in the denominator. Range [0,1] up to roundoff.
- SMTL: `triplet + alpha*separation + beta*concentration + gamma*alignment`.
  Defaults are `alpha=1`, `beta=0.5`, `gamma=0.3`; each is nonnegative. Setting
  weights to zero disables the corresponding optional supervision requirements.

The spectral basis is detached while the field and query remain differentiable.
This is a **fixed-graph adaptation**, not full differentiation through SFM's
manifold-dependent graph. `fixed_spectral_basis(source)` constructs a dense RBF
cosine graph and normalized Laplacian from training inputs once. This differs
from the upstream kNN graph. It uses the complete basis so both low and high
frequencies are available. Keep its row order paired with the input batch; do not
reuse a basis after shuffling/subsampling rows. Independent minibatches/documents
need their own fixed bases. An external basis must be full and orthonormal with
columns ordered low to high eigenvalue; the API cannot verify row provenance or
frequency order. Spectral batches have 3–512 rows; graph cost is O(N³). Other
losses also cap dense batches at 512 rows.

Eigenvector signs do not affect energy. Rotations within repeated-eigenvalue
subspaces **can** change individual mode entropy or a cutoff through that block.
A fixed basis makes an optimization run consistent, not canonically invariant.
PyTorch documents eigenvector gradient instability near repeated eigenvalues:
https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigh.html

Geometry uses the same parametrizations as the NumPy API: flat product torus in
R4, product metric as root-sum-of-squared factor distances. No sphere fallback is
used for other surfaces. Möbius (also in a product) requires `metric='ambient'`;
`geodesic_triplet` rejects it. Angular cut loci and sphere antipodes are
nondifferentiable; tests check ordinary points and finite zero subgradients at
coincidence, not a unique derivative at every cut locus.

## Train and export

```python
import torch
from manifold_studio import Sphere, Torus
from manifold_studio.training import TrainableProjector
from manifold_studio.losses import make_loss, fixed_spectral_basis

# X: training-only source vectors [N,D]; q: source-space query [1,D].
# triplets: training-only integer indices [T,3], all three indices distinct.
model = TrainableProjector(X.shape[1], Sphere() * Torus())
model.fit_normalization(X)
objective = make_loss('smtl', model.geometry)
U = fixed_spectral_basis(X)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(100):
    optimizer.zero_grad()
    result = objective(model(X), source=X, triplets=triplets,
                       query_parameters=model(q), basis=U)
    result.total.backward()
    optimizer.step()

model.eval().fit_display(X)  # Fit 3D display only after optimization.
embedding = model.transform(X)
embedding.save('embeddings.npz')
embedding.export_csv('coordinates.csv')
embedding.plot_3d(target_index=0).write_html('trained.html')
model.save('trained_projector.npz')  # Non-pickle weights + geometry + normalization.
restored = TrainableProjector.load('trained_projector.npz')
held_out = restored.transform(X_test)  # No fitting or graph rebuilding on test rows.
```

Initialize the model in the same dtype/device as the source tensors, e.g.
`.double()` or `.to(device)`, before normalization. `load()` restores on CPU;
move it explicitly for accelerator inference. Refit display after further
training. The serialized embeddings distinguish `trained MLP + parametrization`
from the existing PCA transform. Model files preserve weights and transforms,
not optimizer state or a resumable training session.

Runnable commands (the default data are synthetic):

```bash
python examples/train_projection.py --loss geodesic_triplet --geometry sphere
python examples/train_projection.py --loss smtl --geometry 'sphere*torus' --alpha 1 --beta 0.5 --gamma 0.3
python examples/train_projection.py --loss ambient_triplet --geometry mobius
python examples/train_projection.py --loss distance_alignment --input train.npy
python examples/train_projection.py --loss smtl --input train.npy --triplets triplets.npy --query query.npy
```

Outputs: `embeddings.npz`, `coordinates.csv`, `trained_projector.npz`, and
`training.json` with configuration and every component per step. No text encoder,
model weights or datasets are downloaded by these examples. They train a small
MLP over provided embeddings, not the original sentence encoder. Missing triplets
or queries are rejected; user data never receive arbitrary synthetic labels.

## Historical audit and intentional repairs

Sources reviewed, with zero-based cell numbers:

1. [2025 triplet implementation](https://github.com/vinitchavan/manifold-embedding-nlp/blob/bd44ad837cbacaa005e694af179234e4c15d0790/train.py),
   `test_notebooks/ag_news.ipynb` cell 6 and `mbti_data_2.ipynb` cell 3:
   squared Euclidean triplet, even when outputs lie on a surface.
2. [SFM losses.py](https://github.com/vinitchavan/spectral-field-memory/blob/58aa2fec5ea6b29d287f832240a2f42da56c88ab/sfm/losses.py):
   forward-only NumPy SMTL; high-frequency fraction over total energy,
   unnormalized entropy and max-normalized distance alignment.
3. [Historical SFM notebook](https://github.com/vinitchavan/spectral-field-memory/blob/06b1f78a62b8b297618ae7b34047f33631bfcd75/notebooks/SFM_Experiments.ipynb),
   and the supplied `SFM_Experiments (16).ipynb`, cells 18 and 22. Cell 18 fixes
   orthogonality-based separation, uses signal+noise energy in the denominator,
   min-max distance alignment and entropy normalized by active modes. The
   package instead uses total energy, zero-preserving max normalization and a
   fixed N-mode entropy denominator, documented above.
4. Historical `SFM_Graph_Filtered_Excitation.ipynb` and
   `SFM_MultiHop_Experiments.ipynb` at the same initial commit were checked for
   additional loss definitions; none were found. They are not bundled.

Cell 22's old PyTorch path still computes orthogonality between disjoint columns
of an orthonormal SVD basis, which is zero by construction. It also uses row
energy instead of spectral-mode energy, sphere distances for every manifold, and
a 768-dimensional query against 128-dimensional projected documents. This package
repairs those issues by using real mode energies, shared query/document projection,
and geometry-specific distances. It deliberately does not copy that cell verbatim.

The Arquin/Arlequin archive is still unavailable after file and history searches.
History mentions anchor, area and curvature ideas, but no verified source was
available to faithfully port those implementations. No private client notebook
or client dataset has been published with this change. The historical research
informs this adaptation; no new mathematical novelty or benchmark superiority is
claimed by the API.
