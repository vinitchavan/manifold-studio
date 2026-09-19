"""Optional Plotly views; rendering is separate from the stored representation."""
from __future__ import annotations

from html import escape
import numpy as np

from .geometry import Product, Plane, Circle, Sphere, Cylinder, Torus, Mobius


def _pad3(x):
    return np.pad(x, ((0, 0), (0, max(0, 3-x.shape[1]))))[:, :3]


def factor_display(geometry, parameters):
    """Torus donut is a non-isometric display; all other factors use ambient xyz."""
    if isinstance(geometry, Torus):
        u, v = parameters.T
        r, R = geometry.radius2, geometry.radius1+2*geometry.radius2
        return np.column_stack([(R+r*np.cos(v))*np.cos(u), (R+r*np.cos(v))*np.sin(u), r*np.sin(v)])
    return _pad3(geometry.from_parameters(parameters))


def factor_display_tangent(geometry, parameters, vectors):
    if not isinstance(geometry, Torus):
        return _pad3(vectors)
    # Push a tangent through the donut DISPLAY map, not through an isometry.
    velocity = np.einsum("nij,nj->ni", np.linalg.pinv(geometry.jacobian(parameters)), vectors)
    u, v = parameters.T
    r, R = geometry.radius2, geometry.radius1+2*geometry.radius2
    du = np.column_stack([-(R+r*np.cos(v))*np.sin(u), (R+r*np.cos(v))*np.cos(u), np.zeros_like(u)])
    dv = np.column_stack([-r*np.sin(v)*np.cos(u), -r*np.sin(v)*np.sin(u), r*np.cos(v)])
    return du*velocity[:, :1] + dv*velocity[:, 1:]


def _surface(go, geometry, parameters):
    if isinstance(geometry, Circle):
        p = np.linspace(0, 2*np.pi, 100)[:, None]
        xyz = factor_display(geometry, p)
        return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
                            line=dict(color="#42657a", width=6), name="Circle", hoverinfo="skip")
    if isinstance(geometry, Plane):
        lo = np.minimum(parameters.min(0), -2)
        hi = np.maximum(parameters.max(0), 2)
        u, v = np.meshgrid(np.linspace(lo[0], hi[0], 25), np.linspace(lo[1], hi[1], 25))
    else:
        if isinstance(geometry, Sphere):
            limits = (-np.pi/2, np.pi/2)
        elif isinstance(geometry, Mobius):
            limits = (-geometry.width, geometry.width)
        elif isinstance(geometry, Cylinder):
            limits = (min(-2, float(parameters[:, 1].min())), max(2, float(parameters[:, 1].max())))
        else:
            limits = (0, 2*np.pi)
        u, v = np.meshgrid(np.linspace(0, 2*np.pi, 55), np.linspace(*limits, 30))
    xyz = factor_display(geometry, np.column_stack([u.ravel(), v.ravel()]))
    return go.Surface(x=xyz[:, 0].reshape(u.shape), y=xyz[:, 1].reshape(u.shape),
                      z=xyz[:, 2].reshape(u.shape), opacity=0.22, showscale=False,
                      colorscale=[[0, "#287184"], [1, "#5dbbb6"]], name="Surface", hoverinfo="skip")


def plot_3d(embedding, labels=None, target_index=None, show_labels=False, title="Manifold Studio"):
    """Return a rotatable Figure with product/factor dropdown and optional arrows.

    Arrows are tangent-projected ambient displacements toward target_index.
    In the full-product PCA view they are the linear images of those tangents.
    The chart is diagnostic, not a lossless 3D representation of the product.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError('Install visualization support with pip install ".[viz]"') from e
    n = len(embedding.coordinates)
    if n > 5000:
        raise ValueError("Interactive plots limited to 5,000 points; sample first")
    if labels is None:
        colors = np.zeros(n, dtype=int)
    else:
        labels = np.asarray(labels, dtype=str)
        if labels.shape != (n,):
            raise ValueError("labels must have one value per point")
        _, colors = np.unique(labels, return_inverse=True)
    texts = [escape(str(t)) for t in embedding.texts]
    g = embedding.geometry
    tangents = None if target_index is None else embedding.toward(target_index)
    views = [("Full product · PCA", embedding.display(), None,
              None if tangents is None else tangents @ embedding.display_components.T,
              f"{g.intrinsic_dim} intrinsic dimensions · {g.ambient_dim} ambient coordinates → 3D PCA (distorted)")]
    blocks = list(g.blocks()) if isinstance(g, Product) else [(g, slice(None), slice(None))]
    for i, (factor, ps, xs) in enumerate(blocks):
        p = embedding.parameters[:, ps]
        detail = "Flat torus in R4; donut is a non-isometric 3D illustration" if isinstance(factor, Torus) else "Ambient surface view"
        if isinstance(factor, Mobius):
            detail += "; intrinsic distance unavailable"
        views.append((f"Factor {i+1} · {type(factor).__name__}", factor_display(factor, p),
                      _surface(go, factor, p), None if tangents is None else factor_display_tangent(factor, p, tangents[:, xs]), detail))
    fig = go.Figure()
    groups, point_indices = [], []
    default = 1 if len(blocks) == 1 else 0
    for view_index, (name, xyz, surface, arrows, detail) in enumerate(views):
        group = []
        if surface is not None:
            group.append(len(fig.data))
            surface.visible = view_index == default
            fig.add_trace(surface)
        group.append(len(fig.data))
        point_indices.append(len(fig.data))
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                                  mode="markers+text" if show_labels else "markers",
                                  text=texts, textposition="top center", textfont=dict(size=10),
                                  marker=dict(size=5, color=colors, colorscale="Turbo", opacity=0.92),
                                  customdata=np.arange(n), hovertemplate="%{text}<br>row %{customdata}<extra></extra>",
                                  name="Embeddings", visible=view_index == default))
        if arrows is not None:
            # Fixed scalar across the view, preserving relative lengths/directions.
            norms = np.linalg.norm(arrows, axis=1)
            scale = 0.5/max(float(np.max(norms)), 1e-12)
            indices = np.flatnonzero(norms > 1e-10)[:150]
            ar = arrows[indices]*scale
            base = xyz[indices]
            group.append(len(fig.data))
            fig.add_trace(go.Cone(x=base[:, 0], y=base[:, 1], z=base[:, 2],
                                  u=ar[:, 0], v=ar[:, 1], w=ar[:, 2], anchor="tail",
                                  sizemode="absolute", sizeref=0.22, showscale=False,
                                  colorscale=[[0, "#f5bd62"], [1, "#f5bd62"]],
                                  name="Projected tangent directions (scaled; up to 150)",
                                  hoverinfo="skip", visible=view_index == default))
            group.append(len(fig.data))
            fig.add_trace(go.Scatter3d(x=[xyz[target_index, 0]], y=[xyz[target_index, 1]], z=[xyz[target_index, 2]],
                                      mode="markers", marker=dict(size=10, color="#fff2ad", symbol="diamond"),
                                      name="Target", hovertext=texts[target_index], visible=view_index == default))
        groups.append(group)
    buttons = []
    for i, (name, _, _, _, detail) in enumerate(views):
        visible = [j in groups[i] for j in range(len(fig.data))]
        buttons.append(dict(label=name, method="update", args=[{"visible": visible},
                       {"title": {"text": escape(title)+"<br><sup>"+detail+"</sup>"}}]))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1422", plot_bgcolor="#0b1422",
                      title=dict(text=escape(title)+"<br><sup>"+views[default][4]+"</sup>", x=0.04),
                      height=740, margin=dict(l=15, r=15, b=45, t=145),
                      scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                      updatemenus=[dict(buttons=buttons, active=default, x=0.02, y=1.11, xanchor="left"),
                                   dict(type="buttons", direction="right", x=0.5, y=1.11,
                                        buttons=[dict(label="Show text", method="restyle", args=[{"mode": "markers+text"}, point_indices]),
                                                 dict(label="Hide text", method="restyle", args=[{"mode": "markers"}, point_indices])])],
                      annotations=[dict(text="Drag to rotate · Scroll to zoom · Hover for text · Switch factor views above",
                                        x=0.5, y=-0.04, xref="paper", yref="paper", showarrow=False, font=dict(color="#a0b5c9"))],
                      legend=dict(x=0.01, y=0.98))
    return fig
