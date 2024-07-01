import pytest
from llm4explore.model.common import ANNSampler, KNNSampler
import numpy as np


@pytest.mark.parametrize(
    "sampler_cls",
    [ANNSampler, KNNSampler],
)
def test_retrieve_correct_knns(sampler_cls):
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
    sampler = sampler_cls(
        knn_embeddings=knn_embeddings, k_min=2, k_max=2, dist_threshold=0.1
    )

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


@pytest.mark.parametrize(
    "sampler_cls",
    [ANNSampler, KNNSampler],
)
def test_check_leakage_knns(sampler_cls):
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
    sampler = sampler_cls(
        knn_embeddings=knn_embeddings,
        k_min=2,
        k_max=10,
        dist_threshold=99,
        check_leakage=True,
    )

    # Define a query vector (present in the knn_embeddings)
    query = np.array([3, 4, 5, 6, 7])

    with pytest.raises(ValueError):
        # Get the k nearest neighbors
        index, dist = sampler.sample(query)


@pytest.mark.parametrize(
    "sampler_cls",
    [ANNSampler, KNNSampler],
)
def test_dist_threshold(sampler_cls):
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
    sampler = sampler_cls(
        knn_embeddings=knn_embeddings, k_min=2, k_max=10, dist_threshold=0.1
    )

    # Define a query vector (present in the knn_embeddings)
    query = np.array([3, 4, 5, 6, 7.01])

    indices, dists = sampler.sample(query)
    assert len(indices) == 2
    assert 2 in indices
    assert 3 in indices


def test_times():
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
    times = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    # Create KNNSampler instance
    sampler = KNNSampler(
        knn_embeddings=knn_embeddings, times=times, k_min=2, k_max=10, dist_threshold=0.1
    )

    # Define a query vector (present in the knn_embeddings)
    query = np.array([3, 4, 5, 6, 7.01])

    indices, dists = sampler.sample(query, time_split=1)
    assert len(indices) == 2
    assert 2 not in indices
    assert 1 in indices
    assert 3 in indices
