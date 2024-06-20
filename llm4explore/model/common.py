"""Common functions for models."""
import asyncio
import json
import os
from typing import Any, Dict, List, Tuple

from annoy import AnnoyIndex
import numpy as np
import hashlib

from ..utils import api


def process_embedding_requests(
    model_name: str,
    data: List[str],
    **kwargs,
) -> np.ndarray:
    if model_name == "text-embedding-ada-002":
        # Write data into a temporary jsonl files
        os.makedirs("tmp", exist_ok=True)
        requests_filepath = "tmp/embedding-requests.jsonl"
        save_filepath = "tmp/text-embedding-ada-002-embeddings.jsonl"
        with open(requests_filepath, "w") as f:
            for i, text in enumerate(data):
                f.write(
                    json.dumps({
                        "model": "text-embedding-ada-002",
                        "input": text,
                        "metadata": {
                            "id": i
                        },
                    }) + "\n")
        # Send requests to the API
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            api.process_api_requests_from_file(
                requests_filepath=requests_filepath,
                save_filepath=save_filepath,
                **kwargs,
            ))
        # Read the embeddings and note the order
        id2embedding = {}
        with open(save_filepath, "r") as f:
            for line in f:
                response = json.loads(line)
                embedding = response[1]["data"][0]["embedding"]
                idx = response[-1]["id"]
                id2embedding[idx] = embedding
        assert len(data) == len(
            id2embedding), f"{len(data)} != {len(id2embedding)}"
        embeddings = np.array([id2embedding[i] for i in range(len(data))])
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return embeddings


def process_chat_requests(
    model_name: str,
    messages: List[List[Dict[str, str]]],
    **kwargs,
) -> List[str]:
    if model_name in {"gpt-35-turbo", "gpt-4"}:
        # Write data into a temporary jsonl files
        os.makedirs("tmp", exist_ok=True)
        requests_filepath = "tmp/chat-requests.jsonl"
        save_filepath = f"tmp/{model_name}-chats.jsonl"
        with open(requests_filepath, "w") as f:
            for i, message in enumerate(messages):
                f.write(
                    json.dumps({
                        "model": model_name,
                        "messages": message,
                        "metadata": {
                            "id": i
                        },
                    }) + "\n")
        # Send requests to the API
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            api.process_api_requests_from_file(
                requests_filepath=requests_filepath,
                save_filepath=save_filepath,
                **kwargs,
            ))
        # Read the chat responses and note the order
        id2chat = {}
        with open(save_filepath, "r") as f:
            for line in f:
                response = json.loads(line)
                chat = response[1]["choices"][0]["message"]["content"]
                idx = response[-1]["id"]
                id2chat[idx] = chat
        assert len(messages) == len(
            id2chat), f"{len(messages)} != {len(id2chat)}"
        chats = [id2chat[i] for i in range(len(messages))]
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return chats


def hash_array(array: np.ndarray) -> str:
    """Hash the array using MD5 and return the hash as a string."""
    return hashlib.md5(array.tobytes()).hexdigest()


def check_tmp_dir(func):
    """Function decorator to check the existence of tmp directory."""

    def wrapper(*args, **kwargs):
        os.makedirs("tmp", exist_ok=True)
        return func(*args, **kwargs)

    return wrapper


class KNNSampler():
    """Sampler for KNN search."""

    def __init__(self,
                 knn_embeddings: np.ndarray,
                 metric: str = "euclidean",
                 n_trees: int = 10,
                 k: int = 5,
                 check_leakage: bool = True):
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
        if self.check_leakage and any(dists[i] < 1e-6
                                      for i in range(len(dists))):
            raise ValueError("Query is too close to a sample."
                             "The samples may contain query itself.")
        return indices, dists
