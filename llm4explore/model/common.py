"""Common functions for models."""
import asyncio
import json
import os
from typing import List

import numpy as np

from ..utils import api


def process_embedding_requests(
    model_name: str,
    data: List[str],
    **kwargs,
) -> np.ndarray:
    if model_name == "text-embedding-ada-002":
        # Write data into a temporary jsonl files
        os.makedirs("tmp", exist_ok=True)
        requests_filepath = "tmp/requests.jsonl"
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
        assert len(data) == len(id2embedding)
        embeddings = np.array([id2embedding[i] for i in range(len(data))])
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return embeddings
