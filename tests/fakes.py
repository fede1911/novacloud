"""Test doubles shared by the tests."""

import re
import zlib

import numpy as np

HASHING_DIMENSIONS = 64


class HashingEmbedder:
    """Offline, deterministic embedder for the tests: bag of words hashed into a fixed-size vector."""

    name = "hashing-test"
    model_id = "hashing-test"
    dimensions = HASHING_DIMENSIONS

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        return np.vstack([self.embed_query(document) for document in documents])

    def embed_query(self, query: str) -> np.ndarray:
        vector = np.zeros(HASHING_DIMENSIONS)
        for word in re.findall(r"\w+", query.lower()):
            vector[zlib.crc32(word.encode()) % HASHING_DIMENSIONS] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector
