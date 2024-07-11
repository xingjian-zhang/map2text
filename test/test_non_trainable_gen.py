import logging
import os
from typing import List

import numpy as np
import pytest

from llm4explore.utils.api import process_embedding_requests
from llm4explore.model.non_trainable_gen import (
    EmbeddingBasedGenerator,
    PlagiarismGenerator,
    PromptingBasedGenerator,
)


class TestNonTrainableGen:
    @pytest.fixture
    def data_old(self) -> List[str]:
        return ["Hello World", "Goodbye World", "What is the meaning of life?"]

    @pytest.fixture
    def low_dim_embeddings_old(self) -> np.ndarray:
        return np.array([[1, 2], [3, 4], [99, 99]])

    @pytest.fixture
    def high_dim_embeddings_old(self) -> np.ndarray:
        embeddings = process_embedding_requests(
            "text-embedding-ada-002",
            ["Hello World", "Goodbye World", "What is the meaning of life?"],
            request_url=https://api.openai.com/v1/embeddings,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_requests_per_minute=100,
            max_tokens_per_minute=1000,
            token_encoding_name="cl100k_base",
            max_attempts=5,
            logging_level=logging.INFO,
        )
        return embeddings

    @pytest.fixture
    def sampler_kwargs(self):
        return {
            "n_trees": 2,
            "k": 2,
        }

    @pytest.mark.parametrize("weighted", [True, False])
    def test_embedding_based_generator(
        self,
        data_old,
        low_dim_embeddings_old,
        high_dim_embeddings_old,
        weighted,
        sampler_kwargs,
    ):
        generator = EmbeddingBasedGenerator(
            n_dims=2,
            data_old=data_old,
            low_dim_embeddings_old=low_dim_embeddings_old,
            high_dim_embeddings_old=high_dim_embeddings_old,
            weighted=weighted,
            sampler_kwargs=sampler_kwargs,
        )
        generator.vec2text_corrector.accelerator = "cpu"
        results = generator.decode_all(np.array(([[2, 3]])))
        assert len(results) == 1
        print(results)

    def test_plagiarism_generator(
        self, data_old, low_dim_embeddings_old, sampler_kwargs
    ):
        generator = PlagiarismGenerator(
            n_dims=2,
            data_old=data_old,
            low_dim_embeddings_old=low_dim_embeddings_old,
            sampler_kwargs=sampler_kwargs,
        )
        results = generator.decode_all(np.array([[2.5, 3]]))
        assert results[0][0] == "Goodbye World"

    def test_prompting_based_generator(
        self, data_old, low_dim_embeddings_old, sampler_kwargs
    ):
        generator = PromptingBasedGenerator(
            model_name="gpt-3.5-turbo",
            prompt_type="zero-shot-prompting",
            n_dims=2,
            data_old=data_old,
            low_dim_embeddings_old=low_dim_embeddings_old,
            sampler_kwargs=sampler_kwargs,
            api_kwargs=dict(
                request_url="https://api.openai.com/v1/chat/completions",
                api_key=os.getenv("OPENAI_API_KEY"),
                max_requests_per_minute=100,
                max_tokens_per_minute=1000,
                token_encoding_name="cl100k_base",
                max_attempts=5,
                logging_level=logging.INFO,
            ),
        )
        results = generator.decode_all(np.array([[2, 3]]))
        assert len(results) == 1
        print(results)
