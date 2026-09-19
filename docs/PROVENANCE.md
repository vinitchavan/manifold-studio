# Research provenance and scope

Initial implementation: 19 September 2026.

The package is a new NumPy geometry implementation informed by review of:

- https://github.com/vinitchavan/manifold-embedding-nlp at
  `bd44ad837cbacaa005e694af179234e4c15d0790`, including `ag_news.ipynb`
  and `mbti_data_2.ipynb`.
- https://github.com/vinitchavan/spectral-field-memory at
  `58aa2fec5ea6b29d287f832240a2f42da56c88ab`.
- Vinit's uploaded `SFM_Experiments (16).ipynb`, particularly projection-head,
  diagnostics and prefix-memory sections.

No client notebook collection or private Arlequin AI code has been bundled.
The exact Arquin/Arlequin project folder remains unavailable. These notebooks
are experimental references, not dependencies or reproduced benchmark evidence.

Decisions informed by inspection:

1. Fit input transforms once; do not rescale every query independently.
2. Keep source embeddings, manifold coordinates and display positions separate.
3. Define the flat torus product metric explicitly; a donut rendering is not an
   isometric embedding of that metric in R3.
4. Reject unsupported intrinsic Möbius distance instead of labeling ambient
   chord distance “geodesic.”
5. Keep label-supervised fitting out of held-out data.
6. Preserve high-dimensional product coordinates; 3D plots are views.

This release uses standard differential geometry, PCA, and product metrics.
It makes no novelty claim for these mathematical operations. Research on trained
semantic transport, topology discovery, CSMM selection or spectral memory needs
separate implementation and experiments.

Useful upstream documentation:

- NumPy serialization: https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html
- Plotly 3D surfaces: https://plotly.com/python/3d-surface-plots/
- Sentence Transformers: https://www.sbert.net/docs/package_reference/sentence_transformer/model.html
- Geomstats: https://geomstats.github.io/
- Geoopt: https://github.com/geoopt/geoopt

Optional dependencies carry their own licenses; the generated standalone demo
contains Plotly's JavaScript distribution and its license notices. The package's
MIT license covers the new package source, not externally downloaded model weights.

## v0.2 loss integration

See [LOSSES.md](LOSSES.md) for the source-by-source audit, historical commit links,
formula changes and exclusions. Optional training code adapts the original
triplet and corrected SFM objectives. It does not claim the old notebook results.
