"""Common functions for models."""

from dataclasses import dataclass
import hashlib
import os
from typing import List, Tuple
from warnings import warn

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


class ANNSampler:
    """Sampler for KNN search."""

    def __init__(
        self,
        knn_embeddings: np.ndarray,
        metric: str = "euclidean",
        n_trees: int = 10,
        k_min: int = 2,
        k_max: int = 20,
        dist_threshold: float = 0.1,
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

        self.cache_path = f"tmp/knn_cache_{hash_str}_{metric}_{n_trees}.ann"
        self.index = AnnoyIndex(dim, metric)
        self.knn_embeddings = knn_embeddings
        self.metric = metric
        self.n_trees = n_trees
        self.k_min = k_min
        self.k_max = k_max
        self.dist_threshold = dist_threshold
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
            self.k_max,
            include_distances=True,
        )
        if self.check_leakage and any(dists[i] < 1e-6 for i in range(len(dists))):
            raise ValueError(
                "Query is too close to a sample."
                "The samples may contain query itself."
            )
        # Sort by distance, keep the ones with distance < dist_threshold
        # If there are not enough samples, use the k_min closest ones
        sorted_indices = np.argsort(dists)
        indices = np.array([indices[i] for i in sorted_indices])
        dists = np.array([dists[i] for i in sorted_indices])
        mask = dists < self.dist_threshold
        if np.sum(mask) < self.k_min:
            mask = np.zeros_like(mask)
            mask[: self.k_min] = 1
        indices = indices[mask]
        dists = dists[mask]

        return indices, dists


class KNNSampler:
    def __init__(
        self,
        knn_embeddings: np.ndarray,
        times: np.ndarray = None,
        metric: str = "euclidean",
        n_trees: int = 10,
        k_min: int = 2,
        k_max: int = 20,
        dist_threshold: float = 0.1,
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
        if dim > 5:
            warn(
                "The dimension of the embeddings is greater than 5."
                "This may lead to a high computational cost."
            )

        self.knn_embeddings = knn_embeddings
        self.times = times
        self.metric = metric
        self.n_trees = n_trees
        self.k_min = k_min
        self.k_max = k_max
        self.dist_threshold = dist_threshold
        self.check_leakage = check_leakage

    def sample(
        self, query: np.ndarray, time_split: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get k nearest neighbors of the query before the time split.

        Args:
            query (np.ndarray): The query vector.
            time_split (int): The time split.

        Returns:
            The indices and distances of the k nearest neighbors.
        """
        # query: (dim,)
        # self.knn_embeddings: (n, dim)

        # Compute the distance between the query and all embeddings
        if time_split is None:
            embeddings = self.knn_embeddings
        else:
            time_mask = self.times < time_split
            embeddings = self.knn_embeddings[time_mask]
        if self.metric == "euclidean":
            dists = np.linalg.norm(embeddings - query[None, :], axis=1)
        elif self.metric == "manhattan":
            dists = np.sum(np.abs(embeddings - query[None, :]), axis=1)
        else:
            raise ValueError("Invalid metric.")

        if self.check_leakage and any(dists < 1e-6):
            raise ValueError(
                "Query is too close to a sample."
                "The samples may contain query itself."
            )

        if self.k_max >= len(dists):
            indices = np.arange(len(dists))
        else:
            indices = np.argpartition(dists, self.k_max)[: self.k_max]
        dists = dists[indices]
        sorted_indices = np.argsort(dists)
        indices = indices[sorted_indices]
        dists = dists[sorted_indices]
        mask = dists < self.dist_threshold
        if np.sum(mask) < self.k_min:
            mask = np.zeros_like(mask)
            mask[: self.k_min] = 1
        indices = indices[mask]
        dists = dists[mask]

        if time_split is not None:
            indices = np.arange(len(self.knn_embeddings))[time_mask][indices]

        return indices, dists


@dataclass
class QRTask:
    """Query-Reference task for idea generation."""

    query_vec: np.ndarray  # (n_dims,)
    query_text: str
    query_time: int
    references_vecs: List[np.ndarray]  # (n_refs, (n_dims,))
    references_texts: List[str]
    references_times: List[int]
