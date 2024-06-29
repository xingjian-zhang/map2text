from typing import Any, Dict, List, Tuple
import datasets
import numpy as np
from transformers import AutoTokenizer

from llm4explore.model.common import KNNSampler


def make_input(
    query: np.ndarray,
    reference_embeddings: np.ndarray = None,
    reference_texts: List[str] = None,
    instruction: str = "Convert the coordinate to text",
    split: str = "|",
) -> str:
    np.set_printoptions(precision=4)
    # Example:
    # Convert the coordinate to text: [1, 2] | [3, 4] reference_text_1 | [1, 4]
    # reference_text_2
    prompts = [f"{instruction}: {query}"]
    if reference_embeddings is not None:
        assert reference_texts is not None
        assert len(reference_embeddings) == len(reference_texts)
        for i in range(len(reference_embeddings)):
            prompts.append(f"{reference_embeddings[i]} {reference_texts[i]}")
    return split.join(prompts)


def get_dataset(
    texts: List[str],
    times: List[int],
    low_dim_embeddings: np.ndarray,
    time_train: Tuple[int, int],
    time_val: Tuple[int, int],
    time_test: Tuple[int, int],
    use_sampler: bool = False,
    sampler_kwargs: Dict[str, Any] = None,
    input_kwargs: Dict[str, Any] = None,
) -> datasets.DatasetDict:
    times = np.array(times)
    train_mask = (times >= time_train[0]) & (times < time_train[1])
    val_mask = (times >= time_val[0]) & (times < time_val[1])
    test_mask = (times >= time_test[0]) & (times < time_test[1])

    if use_sampler:
        sampler = KNNSampler(low_dim_embeddings, times, **sampler_kwargs)
    else:
        sampler = None

    def get_data_split(mask):
        inputs = []
        targets = []
        for i in range(len(texts)):
            if mask[i]:
                if sampler is not None:
                    time = times[i]
                    indices, dists = sampler.sample(low_dim_embeddings[i], time)
                    reference_embeddings = low_dim_embeddings[indices, :]
                    reference_texts = [texts[j] for j in indices]
                    inputs.append(
                        make_input(
                            low_dim_embeddings[i],
                            reference_embeddings=reference_embeddings,
                            reference_texts=reference_texts,
                            **input_kwargs,
                        )
                    )
                else:
                    inputs.append(make_input(low_dim_embeddings[i], **input_kwargs))
                targets.append(texts[i])

        return datasets.Dataset.from_dict({"text": inputs, "target": targets})

    train_dataset = get_data_split(train_mask)
    val_dataset = get_data_split(val_mask)
    test_dataset = get_data_split(test_mask)

    return tokenize_dataset(
        datasets.DatasetDict(
            {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
        )
    )


def tokenize_dataset(
    ds: datasets.DatasetDict, tokenizer: AutoTokenizer
) -> datasets.DatasetDict:
    ds = ds.map(
        lambda x: tokenizer(x["text"], truncation=True, padding=True, max_length=256)
    ).map(
        lambda x: {
            "label": tokenizer(
                x["target"], truncation=True, padding=True, max_length=256
            )["input_ids"]
        }
    )
    return ds
