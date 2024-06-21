import logging
import os

from llm4explore.utils.api import (process_chat_requests,
                                   process_embedding_requests)

from utils import change_dir


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


def test_process_chat_requests(change_dir):
    test_data = [
        [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "What is the weather like today?"
        }],
        [{
            "role": "system",
            "content": "You are a smart assistant."
        }, {
            "role": "user",
            "content": "What is the meaning of life?"
        }],
    ]
    responses = process_chat_requests(
        "gpt-35-turbo",
        test_data,
        request_url="https://embedding-api.openai.azure.com/openai/deployments/"
        "gpt-35-turbo/chat/completions?api-version=2023-12-01-preview",
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        max_requests_per_minute=100,
        max_tokens_per_minute=1000,
        token_encoding_name="cl100k_base",
        max_attempts=5,
        logging_level=logging.INFO,
    )
    assert len(responses) == 2
