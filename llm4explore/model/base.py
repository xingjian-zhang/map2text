"""Base classes for models in the scientific innovation space mapping and generation project.

This module provides abstract base classes for the IdeaMapper and IdeaGenerator,
which are central to creating a low-dimensional representation of scientific key ideas
and generating new ideas from such representations, respectively.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import tqdm


class IdeaMapper(ABC):
    """Abstract base class for mapping key ideas into a low-dimensional space.

    This class is responsible for transforming a list of key ideas into a low-dimensional
    space, facilitating the exploration and mapping of the scientific innovation space.
    """

    def __init__(
        self,
        n_dims: int,
        save_path: str,
    ):
        """Initializes the mapper with the specified dimensions and save path.

        Args:
            n_dims: The number of dimensions of the low-dimensional representation.
            save_path: The path where the encoded representations are to be saved.
        """
        self.n_dims = n_dims
        self.save_path = save_path
        self.model = None

    def encode_and_save(
        self,
        data: List[str],
    ) -> np.ndarray:
        """Encodes a list of key ideas and saves the representations to a file.

        Args:
            data: A list of key ideas to be encoded.

        Returns:
            A numpy array of encoded representations.

        Raises:
            ValueError: If the encoded dimensions do not match the expected dimensions.
        """
        encodings = self.encode_all(data)
        if encodings.shape[1] != self.n_dims:
            raise ValueError(
                f"Expected {self.n_dims} dimensions, got {encodings.shape[1]}."
            )
        self.save(encodings)
        return encodings

    @abstractmethod
    def encode_all(self, data: List[str]) -> np.ndarray:
        """Abstract method to encode a list of key ideas into a low-dimensional space.

        Args:
            data: A list of key ideas to encode.

        Returns:
            A numpy array of low-dimensional representations.
        """
        pass

    def save(self, encodings: np.ndarray):
        """Saves the encoded representations to a numpy file.

        Args:
            encodings: The numpy array of encoded representations to save.
        """
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        np.save(self.save_path, encodings)
        return


class IdeaGenerator(ABC):
    """Abstract base class for generating new key ideas from low-dimensional representations.

    This class is responsible for reconstructing or generating new key ideas from
    low-dimensional representations, typically derived from a reference set of older ideas.
    """

    def __init__(
        self,
        n_dims: int,
        data_old: List[str],
        low_dim_embeddings_old: np.ndarray,
    ):
        """Initializes the generator with dimensions and reference data.

        Args:
            n_dims: The number of dimensions of the low-dimensional representation.
            data_old: A list of older key ideas as a reference for idea generation.
            encodings_old: Low-dimensional representations of the older key ideas.
        """
        self.n_dims = n_dims
        self.data_old = data_old
        self.low_dim_embeddings_old = low_dim_embeddings_old

    @abstractmethod
    def decode(self, low_dim_embedding: np.ndarray) -> Tuple[str, Any]:
        """Abstract method to decode a single low-dimensional representation into a key idea.

        Args:
            encoding: A low-dimensional representation of a key idea.

        Returns:
            The reconstructed or generated key idea as a string.
            And the logging information (JSON serializable).
        """
        pass

    def decode_all(self,
                   low_dim_embeddings: np.ndarray) -> List[Tuple[str, Any]]:
        """Decodes a list of low-dimensional representations into key ideas.

        Args:
            encodings: An array of low-dimensional representations to decode.

        Returns:
            A list of reconstructed or generated key ideas.

        Raises:
            ValueError: If the encoded dimensions do not match the expected dimensions.
        """
        if low_dim_embeddings.shape[1] != self.n_dims:
            raise ValueError(
                f"Expected {self.n_dims} dimensions, got {low_dim_embeddings.shape[1]}."
            )
        return [
            self.decode(encoding) for encoding in tqdm.tqdm(low_dim_embeddings)
        ]
