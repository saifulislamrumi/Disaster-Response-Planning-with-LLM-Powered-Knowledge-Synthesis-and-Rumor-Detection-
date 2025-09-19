"""Vector database utilities built on top of FAISS.

This module wraps the FAISS index to allow storing and querying
documents represented by sentence embeddings. Embeddings are
generated using the ``sentence-transformers`` library. Metadata for
each document is stored alongside the vectors to allow retrieval of
additional information (such as the article URL) when results are
returned.
"""

from __future__ import annotations

import os
import pickle
from typing import List, Dict, Any, Optional

try:
    import numpy as np  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore
except ImportError:
    # Define stubs for unavailable modules so the rest of the code can run
    SentenceTransformer = None  # type: ignore
    faiss = None  # type: ignore
    np = None  # type: ignore

from .config import CONFIG


class VectorDB:
    """Semantic search using FAISS and sentence embeddings with graceful fallbacks."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: Optional[str] = None):
        """Initialise the vector database.

        Parameters
        ----------
        model_name: str
            The name of the sentence-transformer model used to generate
            embeddings.
        index_path: Optional[str]
            Optional path to a persisted FAISS index. If provided and the
            file exists, the index and associated metadata will be loaded
            from disk.
        """
        self.use_faiss = SentenceTransformer is not None and faiss is not None and np is not None
        self.model_name = model_name
        if self.use_faiss:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None
        self.index_path = index_path or CONFIG.faiss_index_path
        self.metadata: List[Dict[str, Any]] = []
        self.index: Optional[Any] = None
        self.dimension: Optional[int] = None
        # Attempt to load existing index and metadata if using FAISS
        if self.use_faiss:
            self._load_index()

    def _load_index(self) -> None:
        """Load the FAISS index and metadata from disk if available."""
        if not self.use_faiss:
            return
        meta_path = f"{self.index_path}.meta"
        if os.path.exists(self.index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(self.index_path)  # type: ignore
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            if self.index.ntotal > 0:
                self.dimension = self.index.d

    def _save_index(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        if not self.use_faiss:
            return
        if self.index is None or self.dimension is None:
            return
        faiss.write_index(self.index, self.index_path)  # type: ignore
        meta_path = f"{self.index_path}.meta"
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def _ensure_index(self, embedding_dim: int) -> None:
        """Ensure that the FAISS index is initialised with the given dimension."""
        if not self.use_faiss:
            return
        if self.index is None:
            # Use IndexFlatL2 for simplicity; can be replaced with other indices
            self.index = faiss.IndexFlatL2(embedding_dim)  # type: ignore
            self.dimension = embedding_dim
        elif self.dimension != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: existing index dimension {self.dimension} != new {embedding_dim}"
            )

    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Add a list of documents and their metadata to the index.

        When FAISS and sentence-transformers are unavailable, this
        function simply appends the documents and their metadata to an
        internal list without computing embeddings. In this mode, the
        index cannot be queried, but contexts are cached for potential
        inspection.
        """
        if len(documents) != len(metadata):
            raise ValueError("Documents and metadata must have the same length")
        if self.use_faiss:
            # Generate embeddings
            embeddings = self.model.encode(documents, convert_to_numpy=True)  # type: ignore
            # Initialise or validate index
            self._ensure_index(embeddings.shape[1])
            # Add vectors to index
            self.index.add(embeddings.astype(np.float32))  # type: ignore
            # Append metadata
            self.metadata.extend(metadata)
            # Save to disk
            self._save_index()
        else:
            # Without FAISS, store metadata and text for potential manual lookup
            for doc, meta in zip(documents, metadata):
                meta = meta.copy()
                meta.setdefault("text", doc)
                self.metadata.append(meta)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the index for the most similar documents to a query.

        If FAISS or the embedding model is not available, this function
        returns an empty list because semantic search cannot be performed.
        """
        if not self.use_faiss or self.index is None or self.index.ntotal == 0:
            return []
        query_embedding = self.model.encode([query], convert_to_numpy=True)  # type: ignore
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)  # type: ignore
        results: List[Dict[str, Any]] = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            result = {
                "distance": float(distance),
                "metadata": self.metadata[idx],
            }
            results.append(result)
        return results