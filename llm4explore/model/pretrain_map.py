"""
Generate k-dimensional embeddings using pre-trained models and dimension
reduction algorithm.

Supported pre-trained models:
- `text-embedding-ada-002`

Supported dimension reduction algorithms:
- `LargeVis`
"""
from typing import Any, Dict, List

import numpy as np

from ..utils import api
from .base import IdeaMapper


class PLMMapper(IdeaMapper):
    def __init__(
        self,
        n_dims: int,
        save_path: str,
        plm_encoder: str,
        dr_algorithm: str,
        plm_kwargs: Dict[str, Any] = None,
        dr_kwargs: Dict[str, Any] = None,
    ):
        super().__init__(n_dims, save_path)
        self.plm_encoder = plm_encoder
        self.dr_algorithm = dr_algorithm
        self.plm_kwargs = plm_kwargs or {}
        self.dr_kwargs = dr_kwargs or {}

    def encode_by_plm(self, data: List[str]) -> np.ndarray:
        """Encode a list of key ideas using a PLM.

        Args:
            data: A list of key ideas to encode.

        Returns:
            A numpy array of encoded representations.
        """
        