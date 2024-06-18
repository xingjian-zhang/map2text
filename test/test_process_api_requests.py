import logging
import os
import tempfile

import pytest

from llm4explore.model.common import process_embedding_requests


@pytest.fixture
def change_dir(tmp_path):
    # Remember the current working directory
    old_dir = os.getcwd()
    # Change to the temporary directory
    os.chdir(tmp_path)
    yield
    # Change back to the original directory
    os.chdir(old_dir)


def test_process_embedding_requests(change_dir):
    test_data = ["test1", "test2", "test3"]
    embeddings = process_embedding_requests(
        "text-embedding-ada-002",
        test_data,
        request_url="https://embedding-api.openai.azure.com/openai/deployments/"
        "text-embedding-api/embeddings?api-version=2023-12-01-preview",
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        max_requests_per_minute=100,
        max_tokens_per_minute=1000,
        token_encoding_name="cl100k_base",
        max_attempts=5,
        logging_level=logging.INFO,
    )
    assert embeddings.shape == (3, 1536)
