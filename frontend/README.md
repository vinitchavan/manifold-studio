# Frontend playground

A responsive HTML/CSS/JavaScript UI served by the FastAPI backend. No Node build
step, CDN scripts or separate frontend server are needed. Start the backend from
the repository root and open http://127.0.0.1:8000.

- Synthetic demo, sentence input (optional encoder), JSON/CSV numeric uploads.
- One to three geometry factors; intrinsic/ambient metric selection.
- PCA mapping or learned projection; seven loss options and SMTL weights.
- Rotatable Plotly 3D graph, per-factor surface dropdown, point text, tangent arrows.
- Neighborhood preservation and training-component curves.
- Export of the last successful configuration as a ZIP through the API.

`app.js` owns form state and API interaction; all geometry and learning remain in
the Python package. The run request is only submitted when the user clicks Run.
Controls remain editable while a run finishes; downloads use the last completed
configuration, not any unsaved form edits. A failed run preserves the prior result.

CSV input must be numeric without headers. JSON must be an array of numeric rows.
Use the package directly for NPY/NPZ inputs or larger batches. Individual points
represent input rows/sentences; contextual token embeddings are a future feature.
