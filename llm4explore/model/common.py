"""Common functions for models."""

import hashlib
import os
from typing import Tuple

import numpy as np
from annoy import AnnoyIndex


def hash_array(array: np.ndarray) -> str:
    """Hash the array using MD5 and return the hash as a string."""
    return hashlib.md5(array.tobytes()).hexdigest()


def check_tmp_dir(func):
    """Function decorator to check the existence of tmp directory."""

    def wrapper(*args, **kwargs):
        os.makedirs("tmp", exist_ok=True)
        return func(*args, **kwargs)

    return wrapper


class KNNSampler:
    """Sampler for KNN search."""

    def __init__(
        self,
        knn_embeddings: np.ndarray,
        metric: str = "euclidean",
        n_trees: int = 10,
        k: int = 5,
        check_leakage: bool = True,
    ):
        """Initialize the sampler.

        Args:
            knn_embeddings (np.ndarray): Embeddings to sample from. Usually the
                low-dimensional embeddings.
            metric (str): Distance metric.
            n_trees (int): Number of trees.
            k (int): Number of neighbors to search for.
            check_leakage (bool): Whether to check for leakage.
        """
        assert isinstance(knn_embeddings, np.ndarray)
        dim = knn_embeddings.shape[1]
        hash_str = hash_array(knn_embeddings)

        self.cache_path = f"tmp/knn_cache_{hash_str}_{metric}_{n_trees}_{k}.ann"
        self.index = AnnoyIndex(dim, metric)
        self.knn_embeddings = knn_embeddings
        self.metric = metric
        self.n_trees = n_trees
        self.k = k
        self.check_leakage = check_leakage

        self._init_knn()

    @check_tmp_dir
    def _init_knn(self):
        if os.path.exists(self.cache_path):
            print("Loading KNN index from cache.")
            self.index.load(self.cache_path)
        else:
            print("Building KNN index.")
            for i, embedding in enumerate(self.knn_embeddings):
                self.index.add_item(i, embedding)
            self.index.build(self.n_trees)
            self.index.save(self.cache_path)
        print("KNN index initialized.")

    def sample(self, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get k nearest neighbors of the query.

        Args:
            query (np.ndarray): The query vector.

        Returns:
            The indices and distances of the k nearest neighbors.
        """
        indices, dists = self.index.get_nns_by_vector(
            query,
            self.k,
            include_distances=True,
        )
        if self.check_leakage and any(dists[i] < 1e-6 for i in range(len(dists))):
            raise ValueError(
                "Query is too close to a sample."
                "The samples may contain query itself."
            )
        return indices, dists
