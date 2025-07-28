from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def encode(texts: list[str]) -> np.ndarray:
    """
    Encode list of texts into embeddings.
    """
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False
    )
    return embeddings