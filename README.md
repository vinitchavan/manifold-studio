# Manifold Studio

**Compose geometries. Inspect embeddings. Keep the full representation.**

An initial Python research package by Vinit K. Chavan. Create Cartesian products
such as `Sphere() * Torus() * Plane()`, map your vectors with a fitted transform,
inspect 3D product/factor views and tangent directions, and export numerical data.

The repository includes a local FastAPI web playground. The core package adds
optional PyTorch projection-head training and selectable research losses to the
NumPy geometry and interactive plotting foundation. The package has not been
published on PyPI.

## Why we created Manifold Studio

**What changes when we change the space in which an embedding lives?**

A sentence encoder gives us a vector. We usually compare those vectors with
cosine similarity or Euclidean distance, but geometry is also a modeling choice:
a sphere emphasizes angular relationships, periodic coordinates introduce wraparound,
and a product lets different factors coexist in one representation. We want to
make those choices visible, programmable, and testable.

Manifold Studio grew out of Vinit K. Chavan's manifold-constrained sentence
embedding and Spectral Field Memory research. Experiments spread across notebooks
made it difficult to reuse a projection, compare losses consistently, explain
what a surface plot meant, or export a representation for another application.
This repository brings those pieces into a shared Python package and an
interactive workbench.

Our goal is to let a student explore geometry, a researcher reproduce an
experiment, and an embedding engineer evaluate a representation using the same
underlying code. A compelling plot is a starting point; preserved neighborhoods,
held-out task performance and reproducibility determine whether a method is useful.

## Research papers and original implementations

| Research | Paper | Original code | Connection to this project |
|---|---|---|---|
| **Manifold-Constrained Sentence Embeddings via Triplet Loss: Projecting Semantics onto Spheres, Tori, and Möbius Strips** — Vinit K. Chavan, 2025 | [arXiv:2505.00014](https://arxiv.org/abs/2505.00014) · [PDF](https://arxiv.org/pdf/2505.00014) | [manifold-embedding-nlp](https://github.com/vinitchavan/manifold-embedding-nlp) | Manifold projection and triplet-learning experiments motivating reusable geometry and training interfaces |
| **Spectral Field Memory (SFM)** | [DOI:10.5281/zenodo.19159692](https://doi.org/10.5281/zenodo.19159692) · [Zenodo record](https://zenodo.org/records/19159692) | [spectral-field-memory](https://github.com/vinitchavan/spectral-field-memory) | Graph/spectral memory research motivating query-conditioned spectral objectives and future memory diagnostics |

The SFM DOI above is supplied by the author; its record could not be independently
retrieved during the documentation update. The arXiv title and author were
verified against its record. These research links are separate from the package's
validation: paper results should not be interpreted as reproduced results of
Manifold Studio.

The wider research direction includes **cluster-specific manifold mixtures**:
asking whether different semantic neighborhoods benefit from different geometries.
That is distinct from a Cartesian product shared by all points. Automatic
per-cluster geometry selection is not implemented here.

For implementation lineage, source commits, historical notebook cells and
intentional changes, read [research provenance](docs/PROVENANCE.md) and the
[loss-function audit](docs/LOSSES.md). This package adapts selected research ideas;
it does not bundle private client experiments or claim to reproduce every notebook.

## What you can do today

| Workflow | Available now |
|---|---|
| Bring data | Synthetic demo, numeric embeddings, optional sentence encoding; JSON/CSV uploads in the UI and NPY input through the CLI |
| Choose geometry | Plane, circle, sphere, cylinder, flat torus, Möbius strip and Cartesian products; up to three factors in the UI |
| Map or learn | A fitted PCA-based mapping or an optional trainable PyTorch projection head |
| Choose the objective | Seven loss choices, including triplet variants and a repaired experimental SMTL adaptation |
| Explore | Rotate and zoom 3D views, inspect point text, switch product/factor views, and show projected tangent directions |
| Measure | Neighborhood overlap, loss-component curves, and a separate Python held-out kNN evaluation utility |
| Reuse results | Download full coordinates, fitted/trained transforms, CSV, metrics and an offline interactive HTML plot |
| Extend the project | Add geometries, objectives, evaluation protocols or UI improvements through contributions |

The current app reports **neighborhood preservation and training loss**, not an
automatic semantic-accuracy score. The sentence encoder represents whole input
sentences. Word-level contextual embeddings and semantic edit trajectories remain
research/development goals.

## The theory behind the workbench

### 1. Source vectors, manifold coordinates and display positions

These are three different objects. Given a source embedding $x_i$:

$$
x_i \in \mathbb{R}^{D}, \qquad u_i=f_\theta(x_i)\in\mathbb{R}^{d},
\qquad z_i=\phi_{\mathcal M}(u_i)\in\mathbb{R}^{A}.
$$

Here $u_i$ contains the parameters of a point on a chosen manifold, $\phi$ is its
parametrization, and $z_i$ is the ambient coordinate vector. A separate display
map produces a 3D view. The numerical export preserves the full representation.
In exploration mode, $f$ is a fitted PCA/standardization transform followed by
parameter bounds; in training mode, it is a learned MLP followed by those bounds.
Neither choosing a surface nor fitting PCA proves that language has that topology.

### 2. Geometry changes distance

On a sphere of radius $r$, normalized position vectors $\hat x,\hat y$ have
shortest-path distance $r\arccos(\hat x^\top\hat y)$. A circle uses the shortest
wrapped angular difference. A cylinder combines circular and height distances.
These differ from the straight chord between two ambient points.

For a product $\mathcal M=\mathcal M_1\times\cdots\times\mathcal M_k$:

$$
d_{\mathcal M}(p,q)^2=\sum_{j=1}^{k}d_{\mathcal M_j}(p_j,q_j)^2.
$$

Intrinsic and ambient dimensions add across factors. For example,
`Sphere() * Torus() * Plane()` has six intrinsic dimensions and nine ambient
coordinates. Its three-dimensional picture necessarily loses information.
This product construction does not automatically learn which semantic attribute
belongs to which factor.

### 3. Tangents describe local directions

At a regular parameter point, the parametrization Jacobian $J(u)$ spans the
tangent space. The orthogonal projection of an ambient vector $v$ is
$J(u)J(u)^+v$, where $+$ denotes the pseudoinverse. The implementation handles
sphere tangents directly from its normal, including at coordinate poles.

The displayed arrows project displacements toward a selected point. They help
inspect local directions; they are not learned meanings such as “negate this
sentence,” and are not generic geodesic paths. A true sphere log map is available;
general parallel transport is future work.

### 4. Triplets teach relative relationships

An anchor $a$, positive $p$, and negative $n$ specify which pair should be closer:

$$
\mathcal L_{\mathrm{triplet}}=\frac1T\sum_{t=1}^{T}
\max\left(0,d(a_t,p_t)-d(a_t,n_t)+m\right).
$$

The distance choice matters. Our original-repository adaptation uses squared
ambient distances; intrinsic triplets use supported manifold distances. Margin
$m$ is expressed in the units of that selected objective. Supervision must come
from training data, labels, or an explicitly described mining rule.

### 5. Spectral objectives describe how query energy is distributed

SFM motivates representing a collection as a graph, decomposing its Laplacian
into modes, and measuring where query-conditioned information falls in that
spectrum. In this package's fixed-graph adaptation, $U$ is an orthonormal
Laplacian eigenbasis built from the training source vectors. For projected
query $q$, weights $g_i=\operatorname{softmax}_i(-d(z_i,q)/\tau)$ produce
$F_i=g_i z_i$. Mode energies are

$$
E_k=\left\|(U^\top F)_k\right\|_2^2,\qquad
P_k=\frac{E_k}{\sum_j E_j}.
$$

The separation term penalizes the energy fraction in high-frequency modes;
the concentration term is normalized entropy $-\sum_kP_k\log P_k/\log N$.
Calling high frequencies “noise” is a modeling hypothesis to evaluate, not an
intrinsic property of every text graph. Basis signs do not affect energies;
rotations in repeated-eigenvalue subspaces can affect mode-wise entropy.

Our experimental combined objective is

$$
\mathcal L_{\mathrm{SMTL}}=\mathcal L_{\mathrm{triplet}}
+\alpha\mathcal L_{\mathrm{separation}}
+\beta\mathcal L_{\mathrm{concentration}}
+\gamma\mathcal L_{\mathrm{alignment}}.
$$

Alignment compares normalized manifold and source cosine distances; the
historical name “curvature alignment” does not make it a curvature estimator.
The graph/basis are detached, while gradients flow through the projected field
and query. This is not end-to-end differentiation through graph construction.
See [the exact formulas and API contracts](docs/LOSSES.md) before using these
losses in a paper or benchmark.

### 6. Visualization, topology and useful semantics are separate questions

A Möbius strip is non-orientable, but displaying embeddings on it does not show
that polysemy or negation has Möbius topology. Periodic coordinates may be useful
without proving that language is periodic. Likewise, reduced spectral entropy
can accompany representation collapse. Our synthetic training tutorial actually
shows lower loss alongside worse held-out accuracy than its source baseline.

Evaluate source embeddings, dimension-matched baselines and learned geometries
under independent splits and equal supervision. Use validation data for geometry
and loss selection; reserve test data for the final assessment. See
[validation and limitations](docs/VALIDATION.md).

## Related work and the novelty boundary

Manifold Studio does **not** claim that non-Euclidean embeddings, product
manifolds, complex-valued embeddings, or context-sensitive representations are
new ideas by themselves. Each has substantial prior work:

- **Product / mixed-curvature manifolds.** Gu et al. (ICLR 2019) learned
  representations in products of spherical, hyperbolic and Euclidean model
  spaces and studied how the geometry of the embedding space can be matched to
  heterogeneous data structure
  ([paper](https://openreview.net/forum?id=HJxeWnCcF7)).
- **Context-dependent representations.** Contextual word representations have
  long modeled the fact that the representation of a word changes with its
  linguistic context; ELMo is an important early example
  ([Peters et al., NAACL 2018](https://aclanthology.org/N18-1202/)).
- **Complex-valued embeddings.** Complex word embeddings have been explored in
  NLP, including magnitude/phase-inspired representations
  ([Li et al., 2018](https://aclanthology.org/W18-3006/)), and have since been
  trained at substantially larger scale
  ([Harvey et al., 2024](https://arxiv.org/abs/2412.13745)).
- **Manifold structure inside language models.** Recent work reports shared
  local/global geometry and lower-dimensional structure in language-model token
  embeddings
  ([Lee et al., 2025](https://arxiv.org/abs/2503.21073);
  [Kataiwa et al., 2025](https://arxiv.org/abs/2503.02142)).
- **Context as a transformation, not only a location.** Hu, Niu and Varma
  (2026) explicitly formalize concept representations as point-cloud manifolds
  and contextual transformations as vector fields in language models
  ([paper](https://arxiv.org/abs/2607.04525)).

These results narrow the research question in a useful way. The interesting
problem is no longer simply *whether* language representations can be placed in
curved, product or complex spaces. The harder question is whether we can
**discover which geometry is justified by a semantic phenomenon, learn the
transformations acting on that geometry, and show that those choices improve
generalization, interpretability or memory/retrieval behavior under independent
evaluation**.

Manifold Studio should therefore be read as an experimental workbench and a
starting point for this question—not as a claim that manifold embeddings or
geometric views of contextual meaning were invented here.

## Food for thought: from choosing geometry to discovering it

**What if understanding a word meant learning the transformation induced by
context—not only where the resulting representation sits?**

“Bank” beside a river and “bank” in a financial report provide a simple starting
example. Modern contextual language models already produce different
representations for these uses, and recent research shows that contextual
changes themselves can exhibit structured geometric behavior. The open question
for this project is more specific:

> **Can we identify the geometric operator, topology or collection of factors
> that best explains a semantic transformation, rather than selecting a surface
> first and interpreting the visualization afterward?**

That suggests several falsifiable directions.

### 1. Geometry discovery instead of geometry decoration

Rather than manually choosing `Sphere()`, `Torus()` or `Mobius()`, a future
system could select among candidate geometries using training/validation evidence
while penalizing unnecessary complexity. Conceptually,

$$
\mathcal M^* =
\arg\min_{\mathcal M}
\left[
\mathcal L_{\mathrm{semantic}}
+ \lambda\mathcal L_{\mathrm{geometric}}
+ \beta C(\mathcal M)
\right],
$$

where $C(\mathcal M)$ penalizes excessive geometric complexity. A result is only
interesting if the selected geometry transfers to held-out examples or improves
a downstream criterion relative to dimension-matched baselines.

This is related in spirit to prior mixed-curvature/product-space work, so the
research contribution would have to come from the **semantic setting, topology
family, selection criterion, learned transformations, or empirical finding**—not
from using a product manifold by itself.

### 2. Semantic transformations as operators or flows

Suppose a contextual encoder produces $h(w,c)$ for word/span $w$ in context $c$.
Instead of only asking where two points lie, ask whether a transformation

$$
T_c : h(w,c_1) \mapsto h(w,c_2)
$$

has reusable geometric structure.

Examples include negation, tense, modality, speaker perspective, lexical sense,
or compositional changes. One experiment would test whether an operator estimated
from one collection of words predicts the displacement of unseen words under the
same semantic transformation. This is deliberately stronger than plotting an
arrow between two points: the transformation must generalize.

The vector-field view of contextual transformations in
[Hu et al. (2026)](https://arxiv.org/abs/2607.04525) is especially relevant prior
work. Manifold Studio's opportunity is to test explicit candidate geometries,
intrinsic operators, transport rules and topology selection against that broader
idea rather than claiming the idea of contextual motion itself as new.

### 3. Complex phase as a testable semantic variable

A complex-valued extension could represent a coordinate as

$$
z = r e^{i\theta},
$$

or use several complex coordinates such as $\mathbb C^3$. Complex embeddings
already exist, so simply replacing real coordinates with complex coordinates
would not establish novelty.

A stronger hypothesis is that **phase change itself** carries a reproducible
semantic role. For example, does a transformation such as negation or sense
change produce a stable $\Delta\theta$ across words, contexts or models? Does
phase improve prediction after controlling for parameter count and real-valued
baselines? If not, the phase interpretation should be rejected.

### 4. When would a Möbius topology be justified?

A Möbius strip is non-orientable. Projecting embeddings onto it does not show
that language is non-orientable.

A more meaningful experiment would search for semantic transformation loops whose
latent orientation cannot be represented consistently after transport around the
loop. Evidence of an orientation-reversing cycle would provide a reason to test
a Möbius-like quotient topology; failure to find such evidence would be equally
informative.

This turns Möbius geometry from a chosen visualization into a falsifiable
linguistic/topological hypothesis.

### 5. Beyond independent Cartesian factors

The current product construction

$$
\mathcal M_1 \times \mathcal M_2 \times \cdots \times \mathcal M_k
$$

assumes a clean factorization at the level of the representation. Language may
instead require context-dependent interactions between a base concept and its
local semantic state.

One longer-term mathematical direction is to investigate structures such as
fiber bundles or learned local charts, where context changes how local semantic
coordinates are attached to an underlying concept space. That is a research
proposal, not an implemented feature or established result in this repository.

### What would count as progress?

A visually interesting manifold is not enough. Evidence should include some
combination of:

- independent train/validation/test splits and multi-seed reporting;
- dimension- and parameter-matched Euclidean/PCA baselines;
- neighborhood, retrieval and downstream-task preservation;
- out-of-sample transformation prediction;
- explicit topology/geometry selection without test-set tuning;
- ablations showing whether curvature, topology, phase or factorization is doing
  useful work;
- counterexamples and negative results.

**The long-term question is not “which beautiful surface can hold an
embedding?” It is “which geometric structure is demanded by the transformation
we observe, and can that structure predict something we did not fit?”**

Manifold Studio provides pieces needed to investigate that question: selectable
surfaces, product spaces, losses, tangent views, reusable transforms and
evaluation utilities. Complex-valued coordinates, automatic geometry discovery,
general parallel transport, semantic operators and topology inference are
research directions rather than current capabilities.

This is intentionally an invitation rather than a novelty claim. Bring a
relevant paper, counterexample, alternative geometry, reproducible experiment or
negative result. If a simpler Euclidean model explains the same phenomenon, that
is an important result too.

**If this question interests you, [join the conversation](https://github.com/vinitchavan/manifold-studio/issues)
or [contribute an experiment](CONTRIBUTING.md). Let’s find out together.**

## Open for contributions

**Researchers, students, ML engineers and frontend developers are welcome.**
You can contribute a mathematical correction, a documented experiment, a new
geometry or loss, a reproducibility fix, an accessibility improvement, or a
clearer example. Negative results and well-explained limitations are useful
contributions too.

Start with the [contribution guide](CONTRIBUTING.md). Check existing
[issues](https://github.com/vinitchavan/manifold-studio/issues), propose a focused
change, and submit a pull request. For new research methods, include the formula,
source attribution, assumptions, tests and an evaluation plan. Do not describe a
method as novel or superior solely because its visualizations look convincing.

Good starting tasks: document a geometry edge case, improve mobile graph controls,
add an independent gradient check, or provide a small reproducible held-out
experiment. The package source is distributed under the [MIT license](LICENSE);
external datasets, pretrained weights and dependencies retain their own terms.

## Run the web playground

Clone the repository and enter its root first:

```bash
git clone https://github.com/vinitchavan/manifold-studio.git
cd manifold-studio
python -m pip install -e ".[viz,train]"
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Compose up to three geometries, load vectors or
use the demo, choose mapping/training, inspect 3D surfaces and tangent arrows,
view loss curves, and download your embeddings. Sentence input additionally
requires `.[text]` and downloads MiniLM on first use.

Source folders: [FastAPI backend](backend/README.md) and [frontend UI](frontend/README.md).
This is a local workbench; it is not deployed or configured for public multi-user hosting.

## Choose a loss and train

Seven selectable objectives include the original squared Euclidean triplet,
geodesic/ambient triplets, distance alignment, spectral query-energy separation,
energy concentration, and an experimental repaired SMTL combination.

```bash
python -m pip install -e ".[train,viz]"
python examples/train_projection.py --loss smtl --geometry 'sphere*torus'
```

```python
from manifold_studio import Sphere, Torus
from manifold_studio.losses import available_losses, make_loss
objective = make_loss("smtl", Sphere() * Torus(), alpha=1, beta=0.5, gamma=0.3)
print(available_losses())
```

See [loss choices, formulas, training, and historical audit](docs/LOSSES.md) and
[the training notebook](notebooks/02_selectable_losses.ipynb). Spectral components
use fixed detached graph supervision; these are experimental adaptations, not
reproductions of old benchmark results. Train only on training data and compare
on a held-out set before claiming an improvement.

## Install from this repository

Python 3.10 or newer:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[viz]"
```

The core requires NumPy. Plotly is optional. For the optional sentence encoder:

```bash
python -m pip install -e ".[viz,text]"
```

Do not run `pip install manifold-studio` expecting this code: it has not been
published to a package index. Install from the extracted repository directory.

## Compose two or three geometries

```python
import numpy as np
from manifold_studio import Sphere, Circle, Torus, Plane, Mobius, ManifoldProjector

geometry = Sphere() * Torus() * Plane()
# Also: Sphere() * Circle(), Torus() * Mobius(), Sphere() * Sphere()

X = np.load("your_embeddings.npy", allow_pickle=False)  # items × features
mapper = ManifoldProjector(geometry)
result = mapper.fit_transform(X)

print(geometry.intrinsic_dim)  # 6
print(result.coordinates.shape)  # (number of items, 9)
print(result.display().shape)    # (number of items, 3)

fig = result.plot_3d(target_index=0)
fig.show()
fig.write_html("explorer.html", include_plotlyjs=True)  # self-contained/offline

result.save("embeddings.npz")
result.export_csv("coordinates.csv")
mapper.save("projector.npz")
```

This is a **Cartesian product**, not the vector cross product. Products preserve
factor coordinates. They do not magically fit all information into three
dimensions. The interactive dropdown switches between the product's 3D PCA
display and separate factor surfaces. Hover identifies points, drag rotates,
scroll zooms, and buttons toggle labels. Choose an arrow target in Python with
`target_index`; changing targets by clicking is planned for the application.

## Immediate demo, no model download

```bash
manifold-studio demo --geometry 'sphere*torus*plane' --out outputs/demo
```

Open `outputs/demo/explorer.html`. This uses **synthetic numerical points**, not
language embeddings, and reports neighborhood preservation rather than accuracy.
Outputs include full embeddings, a fitted projector, coordinates CSV, metrics,
and a standalone Plotly visualization.

Map your own matrix and optionally annotate rows:

```bash
manifold-studio map --input embeddings.npy --texts sentences.json \
  --geometry 'sphere*circle' --out outputs/my_run
```

`sentences.json` is a list with one text string per row. For products containing
Möbius, explicitly select ambient distance:

```bash
manifold-studio demo --geometry 'torus*mobius' --metric ambient
```

Use `--no-html` without the Plotly dependency. Module invocation also works:
`python -m manifold_studio.cli demo --no-html`.

## Encode sentences

```python
from manifold_studio import encode_sentences, Sphere, Circle, ManifoldProjector

texts = [
    "I deposited money at the bank.",
    "We sat beside the river bank.",
    "The bridge is safe.",
    "The bridge is not safe.",
    "Maya believes the bridge is not safe.",
    "Maya does not believe the bridge is safe.",
    "The company announced higher earnings.",
    "The football team won the final.",
]
X = encode_sentences(texts)  # first use downloads the model
mapper = ManifoldProjector(Sphere() * Circle())
result = mapper.fit_transform(X, texts=texts)
result.plot_3d(show_labels=True, target_index=2).show()
```

These are sentence representations. Contextual word/span extraction and its
tokenizer alignment are planned; this release does not assign sentence vectors
to individual words. Pass `revision=` to `encode_sentences` to pin model weights
and record that revision in your experiment notes. The optional adapter was not
end-to-end model-tested in the initial build environment; core tests need no LLM.

## Geometry conventions

| Geometry | Intrinsic dimension | Ambient coordinates | Intrinsic distance |
|---|---:|---:|---|
| Plane | 2 | 2 | Euclidean |
| Circle | 1 | 2 | Shortest circular arc |
| Sphere | 2 | 3 | Great-circle distance |
| Cylinder | 2 | 3 | Circle arc × linear height |
| Torus | 2 | 4 | Flat Riemannian product of two circles |
| Möbius strip | 2 | 3 | Not implemented; ambient chord distance is explicit |
| Product | Sum | Sum | Square root of summed squared factor distances |

`Torus()` is mathematically `S¹ × S¹` in R4. Its donut rendering in R3 is
**non-isometric**: visual lengths on the donut are not the flat-torus metric.
The Möbius parametrization respects `(u + 2π, v) ~ (u, -v)`. No globally
consistent normal field or intrinsic Möbius shortest-path solver is claimed.

```python
geometry = Sphere(radius=2) * Circle(radius=0.5)
# Radius changes the corresponding factor's metric scale.
```

## Tangent vectors

```python
directions = result.toward(0)
# Tangent-projected ambient displacements, one at each base point.
# They are not generic geodesic/log-map directions.

random_vectors = np.random.default_rng(0).normal(size=result.shape)
tangents = result.tangent_project(random_vectors)
```

The sphere also exposes a true `log_map` for paired angular coordinates; exactly
antipodal points are rejected because the shortest direction is ambiguous.
General parallel transport, geodesic animation, and learned semantic edit
operators are future work. Plot arrows are scaled and limited to 150 nonzero
directions for readability. Full-product arrows are images under the PCA display
map, not tangent vectors to a fictitious 3D product surface.

## Fit once, transform consistently

The mapper fits PCA on all input columns, standardizes retained PC scores, and
allocates consecutive latent coordinates to the geometry factors. Spherical
latitude and Möbius width use smooth bounded maps. This is an explicit descriptive
mapping—not a discovery that language naturally lies on the selected surface,
and not end-to-end semantic training.

Fit on a reference or training collection. New inputs use the saved transform:

```python
from manifold_studio import ManifoldProjector, Embedding

mapper = ManifoldProjector.load("projector.npz")
new_result = mapper.transform(new_embeddings, texts=new_sentences)
old_result = Embedding.load("embeddings.npz")
```

Mapping an input alone and in a batch gives the same coordinates. The training
display PCA also remains fixed, so changes between views are not caused by
refitting the display. Coordinate comparisons require the same fitted mapper.
Degenerate/low-rank collections are rejected instead of inventing dimensions.
For intrinsic dimension `d`, fitting requires at least `d+1` varied examples,
at least `d` input columns, and sufficient rank.

## Evaluate rather than assume

```python
from manifold_studio import neighbor_preservation, evaluate_knn

print(neighbor_preservation(X, result, k=5, metric="intrinsic"))

report = evaluate_knn(
    X_train, y_train, X_test, y_test,
    geometry=Sphere() * Circle(), k=3,
)
print(report)
```

The labeled probe fits its mapper only on training rows. It compares full source
Euclidean kNN, dimension-matched standardized PCA kNN, and manifold kNN, reporting
accuracy and macro-F1. Caller-provided train/test groups must be independent;
upstream embedding leakage cannot be detected by this package. Do not repeatedly
choose geometry on the test set. This initial probe is not a multi-seed benchmark.

Unlabeled data gets neighborhood overlap, not an invented accuracy score. Dense
pairwise diagnostics are capped at 5,000 rows; holdout evaluation is capped at
25 million pairs. Sparse/ANN execution is on the roadmap.

## Export format

`embeddings.npz` stores source vectors, full manifold coordinates, intrinsic
parameters, row-aligned text, the fixed display PCA, and versioned geometry
metadata. `projector.npz` stores the fitted input transform for future queries.
Both use numeric/string arrays and are loaded with `allow_pickle=False`.
`coordinates.csv` contains ambient coordinates, not just the visible three axes.

The bundle includes the source matrix for inspection and is **not a compressed
memory codec**. File sizes are not token savings. Downstream systems must use the
same transform and an appropriate distance; Euclidean ANN on exported coordinates
does not automatically reproduce intrinsic manifold rankings.

## Repository layout

| Path | Responsibility |
|---|---|
| `src/manifold_studio/geometry.py` | Geometries, products, distances, Jacobians, tangents |
| `src/manifold_studio/embedding.py` | Fitted transforms and portable bundles |
| `src/manifold_studio/visualization.py` | Interactive full-product/factor views |
| `src/manifold_studio/evaluation.py` | Neighborhood and held-out classification probes |
| `src/manifold_studio/encoders.py` | Optional sentence-transformer adapter |
| `src/manifold_studio/cli.py` | Numerical demo and matrix-import command |
| `src/manifold_studio/losses.py` | Selectable objectives and fixed spectral basis |
| `src/manifold_studio/torch_geometry.py` | Differentiable geometry operations |
| `src/manifold_studio/training.py` | Trainable projection head and portable weights |
| `backend/` | FastAPI routes and integration tests |
| `frontend/` | Browser controls, plots and API interaction |
| `examples/` | Runnable numerical and text examples |
| `notebooks/` | Package-first walkthrough |
| `tests/` | Mathematical invariants and integration tests |
| `docs/` | Research provenance and development roadmap |

## Test

```bash
python -m unittest discover -s tests -v
```

Tests cover product dimensions and metrics, finite-difference Jacobians, sphere
tangents including poles, Möbius seam transitions, projector batch invariance,
export/reload/replay, validation, evaluation, and optional Plotly serialization.

## Repository and installation

The project repository is [vinitchavan/manifold-studio](https://github.com/vinitchavan/manifold-studio).
The repository is public. The package has not been published to PyPI; install
from a source checkout:

```bash
git clone https://github.com/vinitchavan/manifold-studio.git
cd manifold-studio
python -m pip install -e '.[viz]'
```

## Citing the research

If this project informs your work, cite the relevant paper linked above and
identify the Manifold Studio commit/version used in your experiments. The software
adapts research methods and does not have a separate archival software DOI yet.

```bibtex
@misc{chavan2025manifold,
  title={Manifold-Constrained Sentence Embeddings via Triplet Loss: Projecting Semantics onto Spheres, Tori, and Möbius Strips},
  author={Chavan, Vinit K.},
  year={2025},
  eprint={2505.00014},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.00014}
}
```

For SFM, use the citation metadata exported by its
[Zenodo record](https://zenodo.org/records/19159692) to preserve the exact deposited
version and author information.
