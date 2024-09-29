import logging
import os

from map2text.model.pretrain_map import PLMMapper


def test_plm_mapper():
    plm_mapper = PLMMapper(
        n_dims=2,
        save_path="tmp/ada2_largevis.npy",
        plm_encoder="text-embedding-ada-002",
        dr_algorithm="largevis",
        plm_kwargs=dict(
            request_url=https://api.openai.com/v1/embeddings,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_requests_per_minute=100,
            max_tokens_per_minute=1000,
            token_encoding_name="cl100k_base",
            max_attempts=5,
            logging_level=logging.INFO,
        ),
        dr_kwargs=dict(threads=8),
    )
    embeddings = plm_mapper.encode_all(["test1", "test2", "test3"])
    assert embeddings.shape == (3, 2)
