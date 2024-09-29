from typing import Dict, List

import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer
import atometric


class CosineSimilarity:
    """Compute cosine similarity between two ordered list of texts."""

    def __init__(self):
        """Initialize the SentenceTransformer model."""
        self.encoder = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

    def get_detailed_instruct(self, query: str) -> str:
        """Generate a detailed instruct for the query."""
        return f"Instruct: Retrieve semantically similar text.\nQuery: {query}"

    def get_embeddings(self, texts: List[str], is_query: bool):
        """Compute embeddings for the given texts."""
        if is_query:
            texts = [self.get_detailed_instruct(query) for query in texts]
        embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ):
        """Compute the cosine similarity between predictions and references."""
        predictions_embeddings = self.get_embeddings(predictions, is_query=True)
        references_embeddings = self.get_embeddings(references, is_query=False)
        # Compute pairwise cosine similarity
        cosine_similarities = []
        for pred, ref in zip(predictions_embeddings, references_embeddings):
            cosine_similarities.append(np.dot(pred, ref))  # Already normalized
        cosine_similarities = np.array(cosine_similarities)
        return {"cosine": float(np.mean(cosine_similarities))}


class Evaluation:
    SUPPORTED_METRICS = {
        "bleu": {"source": "huggingface", "average": True},
        "rouge": {"source": "huggingface", "average": True},
        "bertscore": {"source": "huggingface", "average": False},
        "meteor": {"source": "huggingface", "average": True},
        "bleurt": {
            "source": "huggingface",
            "kwargs": {
                "module_type": "metric",
                "checkpoint": "BLEURT-20-D12",
                "config_name": "BLEURT-20-D12",
            },
            "average": False,
        },
        "cosine": {"source": "custom", "average": True},
        "atometric-i": {"source": "custom", "average": False}, # cs-research-idea
        "atometric-c": {"source": "custom", "average": False}, # cs-research-context
        "atometric-p": {"source": "custom", "average": False}, # persona
    }

    def __init__(self, metric_names: List[str] = None):
        if metric_names is None:
            metric_names = self.SUPPORTED_METRICS.keys()
        self.metric_names = metric_names
        self.metrics = {}

        for metric_name in self.metric_names:
            if metric_name not in self.SUPPORTED_METRICS:
                raise ValueError(f"Unsupported metric: {metric_name}")
            metric_info = self.SUPPORTED_METRICS[metric_name]
            if metric_info["source"] == "huggingface":
                metric_kwargs = metric_info.get("kwargs", {})
                self.metrics[metric_name] = evaluate.load(metric_name, **metric_kwargs)
            elif metric_info["source"] == "custom":
                if metric_name == "cosine":
                    self.metrics[metric_name] = CosineSimilarity()
                elif metric_name == "atometric-i":
                    self.metrics[metric_name] = atometric.Atometric.from_task_type("cs_research_idea")
                elif metric_name == "atometric-c":
                    self.metrics[metric_name] = atometric.Atometric.from_task_type("cs_research_context")
                elif metric_name == "atometric-p":
                    self.metrics[metric_name] = atometric.Atometric.from_task_type("persona")
    def compute(self, predictions: List[str], references: List[str]):
        results = {}
        for metric_name, metric in self.metrics.items():
            print(f"Start {metric_name}.")
            if metric_name == "bertscore":
                results[metric_name] = metric.compute(
                    predictions=predictions, references=references, lang="en"
                )
            else:
                results[metric_name] = metric.compute(
                    predictions=predictions, references=references
                )
            if not self.SUPPORTED_METRICS[metric_name]["average"]:
                for k in results[metric_name]:
                    if isinstance(results[metric_name][k], list):
                        # Some values are not numeric (e.g. bert version)
                        results[metric_name][k] = np.mean(
                            np.array(results[metric_name][k], dtype=float)
                        )
            print(f"Finish {metric_name}.")
        return flatten_and_round(results)


def flatten_and_round(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    # Flatten the results
    flat_results = {}
    for metric_name, metric_results in results.items():
        for sub_metric_name, value in metric_results.items():
            flat_results[f"{metric_name}_{sub_metric_name}"] = value
    # Round the float precision to 4 decimal places
    return round_floats(flat_results)


def round_floats(obj, precision=4):
    """Round the floating point numbers in the object."""
    if isinstance(obj, float):
        return round(float(obj), precision)
    elif isinstance(obj, dict):
        return {key: round_floats(value, precision) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, precision) for item in obj]
    return obj
