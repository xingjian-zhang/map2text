import pytest
from llm4explore.model.common import KNNSampler
import numpy as np


def test_retrieve_correct_knns():
    # Generate 10 hand-crafted vectors
    knn_embeddings = np.array(
        [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9],
            [6, 7, 8, 9, 10],
            [7, 8, 9, 10, 11],
            [8, 9, 10, 11, 12],
            [9, 10, 11, 12, 13],
            [10, 11, 12, 13, 14],
        ]
    )
    # Create KNNSampler instance
    sampler = KNNSampler(knn_embeddings=knn_embeddings, k=2)

    # Define a query vector
    query = np.array([3.5, 4.5, 5.5, 6.5, 7.5])

    # Get the k nearest neighbors
    index, dist = sampler.sample(query)

    # Check if the correct indices are retrieved
    assert len(index) == 2
    assert 2 in index
    assert 3 in index

    # Check if the distances are correct
    assert np.allclose(dist, np.sqrt(0.5**2 * 5))


def test_check_leakage_knns():
    # Generate 10 hand-crafted vectors
    knn_embeddings = np.array(
        [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9],
            [6, 7, 8, 9, 10],
            [7, 8, 9, 10, 11],
            [8, 9, 10, 11, 12],
            [9, 10, 11, 12, 13],
            [10, 11, 12, 13, 14],
        ]
    )
    # Create KNNSampler instance
    sampler = KNNSampler(knn_embeddings=knn_embeddings, k=2, check_leakage=True)

    # Define a query vector (present in the knn_embeddings)
    query = np.array([3, 4, 5, 6, 7])

    with pytest.raises(ValueError):
        # Get the k nearest neighbors
        index, dist = sampler.sample(query)
