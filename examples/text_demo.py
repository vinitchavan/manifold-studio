"""Optional sentence model download; no paid API required."""
from pathlib import Path
from manifold_studio import encode_sentences, Sphere, Circle, ManifoldProjector

texts = [
    "I deposited money at the bank.", "We sat beside the river bank.",
    "The bridge is safe.", "The bridge is not safe.",
    "Maya believes the bridge is not safe.", "Maya does not believe the bridge is safe.",
    "The football team won the final.", "The company reported higher earnings.",
    "The patient needs a follow-up appointment.", "The telescope detected a distant galaxy.",
]

if __name__ == "__main__":
    x = encode_sentences(texts)
    mapper = ManifoldProjector(Sphere()*Circle())
    result = mapper.fit_transform(x, texts=texts)
    out = Path("outputs/text_demo"); out.mkdir(parents=True, exist_ok=True)
    result.save(out/"embeddings.npz")
    mapper.save(out/"projector.npz")
    result.plot_3d(target_index=2).write_html(out/"explorer.html", include_plotlyjs=True)
    print(f"Open {(out/'explorer.html').resolve()}")
