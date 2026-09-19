"""Optional sentence encoding; core geometry requires no model downloads."""


def encode_sentences(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", revision=None):
    """First use downloads a model. Sentence outputs are not contextual token vectors.

    Pass revision to pin a model commit for reproducible experiments.
    """
    if isinstance(texts, str):
        raise ValueError("Pass a list of sentences, not a single string")
    texts = list(texts)
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Supply nonempty sentence strings")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError('Install sentence support with pip install ".[text]"') from e
    model = SentenceTransformer(model_name, revision=revision, trust_remote_code=False)
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors
