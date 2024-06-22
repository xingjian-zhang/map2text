"""
Generate k-dimensional embeddings using pre-trained models and dimension
reduction algorithm.

Supported pre-trained models:
- `text-embedding-ada-002`

Supported dimension reduction algorithms:
- `largevis`
"""

import os
from typing import Any, Dict, List
from subprocess import run

import numpy as np

from llm4explore.model.base import IdeaMapper
from llm4explore.model.common import hash_array
from llm4explore.utils.api import process_embedding_requests


class PLMMapper(IdeaMapper):
    """Pre-trained Language Model (PLM) Encoder based Idea Mapper.

    This class uses a pre-trained language model to encode key ideas and then
    applies a dimension reduction algorithm to map the embeddings to a lower
    dimensional space.
    """

    def __init__(
        self,
        n_dims: int,
        save_path: str,
        plm_encoder: str,
        dr_algorithm: str,
        plm_kwargs: Dict[str, Any] = None,
        dr_kwargs: Dict[str, Any] = None,
    ):
        """Initialize the PLM-based Idea Mapper.

        Args:
            n_dims: The number of dimensions to map the embeddings to.
            save_path: The path to save the embeddings.
            plm_encoder: The pre-trained language model encoder to use.
            dr_algorithm: The dimension reduction algorithm to use.
            plm_kwargs: Additional keyword arguments for the PLM encoder.
            dr_kwargs: Additional keyword arguments for the dimension reduction
            algorithm.
        """
        assert save_path.endswith(".npz"), "Save path must end with .npz"
        super().__init__(n_dims, save_path)
        self.plm_encoder = plm_encoder
        self.dr_algorithm = dr_algorithm
        self.plm_kwargs = plm_kwargs or {}
        self.dr_kwargs = dr_kwargs or {}

    def encode_all(self, data: List[str], return_high_dim=False) -> np.ndarray:
        """Encode a list of key ideas using a PLM and reduce dimensions.

        Args:
            data: A list of key ideas to encode.
            return_high_dim: Whether to also return the high-dimensional embeddings.

        Returns:
            A numpy array of encoded representations with reduced dimensions.
        """
        embeddings = self.encode_by_plm(data)
        np.savez(self.save_path, high_dim_embeddings=embeddings)
        reduced_embeddings = self.reduce_dims(embeddings)
        np.savez(
            self.save_path,
            low_dim_embeddings=reduced_embeddings,
            high_dim_embeddings=embeddings,
        )
        if return_high_dim:
            return reduced_embeddings, embeddings
        return reduced_embeddings

    def encode_by_plm(self, data: List[str]) -> np.ndarray:
        """Encode a list of key ideas using a PLM.

        Args:
            data: A list of key ideas to encode.

        Returns:
            A numpy array of encoded representations.
        """
        if self.plm_encoder == "text-embedding-ada-002":
            embeddings = process_embedding_requests(
                model_name="text-embedding-ada-002",
                data=data,
                **self.plm_kwargs,
            )
        else:
            raise ValueError(f"Unsupported PLM encoder: {self.plm_encoder}")
        return embeddings

    def reduce_dims(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce the dimensions of the embeddings.

        Args:
            embeddings: The embeddings to reduce the dimensions of.

        Returns:
            A numpy array of embeddings with reduced dimensions.
        """
        hash_str = hash_array(embeddings)
        if self.dr_algorithm == "largevis":
            reduced_embeddings = self.run_largevis(embeddings, hash_str)
        else:
            raise ValueError(
                f"Unsupported dimension reduction algorithm: {self.dr_algorithm}"
            )
        return reduced_embeddings

    def run_largevis(self, embeddings: np.ndarray, hash_str: str) -> np.ndarray:
        """Run the LargeVis algorithm on embeddings.

        Args:
            embeddings (np.ndarray): Embeddings to reduce.
            hash_str (str): Hash string used for file naming.

        Returns:
            np.ndarray: Embeddings with reduced dimensions.
        """
        in_path = f"tmp/embeddings_{hash_str}.txt"
        out_path = f"tmp/largevis_{hash_str}.txt"
        np.savetxt(
            in_path,
            embeddings,
            header=f"{embeddings.shape[0]} {embeddings.shape[1]}",
            comments="",
        )

        executable_path = "./llm4explore/external/largevis/Linux/LargeVis"
        if not os.path.exists(executable_path):
            raise FileNotFoundError(
                "LargeVis executable not found. Please compile the LargeVis source code."
            )

        # Define command and parameters for running LargeVis
        command = [
            executable_path,
            "-input",
            in_path,
            "-output",
            out_path,
            "-outdim",
            str(self.n_dims),
        ]
        for key, value in self.dr_kwargs.items():
            command.extend([f"-{key}", str(value)])

        # Run LargeVis using the subprocess module
        process = run(command, check=True)
        if process.returncode != 0:
            raise RuntimeError("LargeVis failed to run.")

        # Load the reduced embeddings from output file
        reduced_embeddings = np.loadtxt(out_path, skiprows=1)

        return reduced_embeddings
