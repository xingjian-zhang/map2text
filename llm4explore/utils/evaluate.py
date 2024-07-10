import json
import logging
import os
from typing import Any, Dict, List

import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer

from llm4explore.utils.api import process_chat_requests


LLM_EVAL_PROMPT = [
    {
        "role": "system",
        "content": 'You are an expert in scientific research and can evaluate the semantic similarity between pairs of key-idea aspect summarizations from scientific papers.   Key idea refers to the main intellectual merit of this paper, often in comparison to the context.  This could normally be a novel idea or solution proposed in this paper that distincts it from what’s already done in literature.  You will be provided with two inputs, and your task is to assign a similarity score on a scale from 1 to 10, where 1 indicates completely dissimilar content and 10 indicates identical content. Before scoring, provide a brief justification for your score.  Structure your response in the following JSON format: \n{\n  "score": score,\n  "justification": "Brief explanation for the score"\n}\n',
    },
    {
        "role": "user",
        "content": "Input 1:The authors create a web platform that supports the work of psychologists, streamlining the collection of relevant information about people with dementia.\nInput 2:Non-pharmacological interventions are the most common and the most effective for people with dementia. Using biographical or personalized materials has been proven to benefit these interventions.",
    },
    {
        "role": "assistant",
        "content": '{\n  "justification": Both inputs are related to people with dementia. However, the first input talks about a web platform to help psychologists collect information, while the second input focuses on the effectiveness of non-pharmacological interventions in dementia care, particularly using biographical or personalized materials. They share a similar context but differ in specific focus and application.", \n  "score": 1\n}',
    },
    {
        "role": "user",
        "content": "Input 1:The authors create a web platform that supports the work of psychologists, streamlining the collection of relevant information about people with dementia.\nInput 2:The paper focuses on non-pharmacological interventions for people with dementia, emphasizing the effectiveness of biographical or personalized materials.",
    },
    {
        "role": "assistant",
        "content": '{\n "justification":Both input discuss only one common key-point: people with dementia. Input 1 discusses a web platform to assist psychologists in collecting information about people with dementia.  Input 2 focuses on non-pharmacological interventions for dementia, particularly biographical or personalized materials.  While both relate to dementia, they address different aspects—one is about a technological tool for data collection, and the other about therapeutic methods.  Their key ideas share some thematic overlap but are fundamentally different in focus and application.", \n  "score": 2\n}',
    },
    {
        "role": "user",
        "content": "Input 1:The authors create a web platform that supports the work of psychologists, streamlining the collection of relevant information about people with dementia.\nInput 2:The web platform make improvements in the collection of meaningful data about a person, and on maintaining awareness of the therapy as a whole.",
    },
    {
        "role": "assistant",
        "content": '{\n  "justification": Both inputs describe two main-points: creating a web platform and improving the collection of data for specific professional use.   Input 1 specifies two more key-points: the platform supports psychologists working with people with dementia, while Input 2 emphasizes improvements in data collection and therapy awareness. The core idea of a web platform facilitating data collection is similar, but the specific focus areas slightly differ.", \n  "score": 5\n}',
    },
    {
        "role": "user",
        "content": "Input 1:The authors create a web platform that supports the work of psychologists, streamlining the collection of relevant information about people with dementia.\nInput 2:The authors propose a web platform to collect and manage biographical materials in a streamlined manner, enabling reminiscence and other biographical cognitive stimulation practices for people with dementia.",
    },
    {
        "role": "assistant",
        "content": '{\n  "justification"::Both inputs describe a web platform aimed at supporting the care of people with dementia by streamlining the collection of relevant information.   While input 2 miss one key-point: the platform is supporting psychologists generally, leading to score of 8. ", \n  "score": 8\n}',
    },
    {
        "role": "user",
        "content": "Input 1:The authors create a web platform that supports the work of psychologists, streamlining the collection of relevant information about people with dementia.\nInput 2:The authors created a web platform to support psychologists by streamlining the collection of relevant biographical information about people with dementia, facilitating reminiscence and other biographical cognitive stimulation practices.",
    },
    {
        "role": "assistant",
        "content": '{\n  "justification":Both inputs describe a web platform designed to assist psychologists by streamlining the collection of information about people with dementia.   Input 2 provides a little additional detail by mentioning biographical information, but the core idea remains nearly identical. ", \n  "score": 9\n}',
    },
]


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


class LLMEval:
    """Compute the LLM evaluation metric."""

    def __init__(self, model_name: str = "gpt-4o", api_kwargs: Dict[str, Any] = None):
        self.model_name = model_name
        if api_kwargs is None:
            api_kwargs = dict(
                request_url="https://api.openai.com/v1/chat/completions",
                api_key=os.getenv("OPENAI_API_KEY"),
                max_requests_per_minute=2000,
                max_tokens_per_minute=200000,
                token_encoding_name="cl100k_base",
                max_attempts=5,
                logging_level=logging.INFO,
            )
        self.api_kwargs = api_kwargs

    def generate_prompt(self, text_1: str, text_2: str):
        """Generate the prompt for the language model."""
        user_prompt = f"Input 1: {text_1}\nInput 2: {text_2}"
        message = LLM_EVAL_PROMPT + [{"role": "user", "content": user_prompt}]
        return message

    def compute(self, predictions: List[str], references: List[str]):
        messages = [
            self.generate_prompt(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        results = process_chat_requests(
            model_name=self.model_name,
            messages=messages,
            parameters={
                "temperature": 1,
                "top_p": 0.95,
                "response_format": {"type": "json_object"},
            },
            **self.api_kwargs,
        )
        results_numeric = []
        for r in results:
            try:
                results_numeric.append(float(json.loads(r)["score"]))
            except Exception as e:
                logging.warning(f"Invalid result: {r}")
                logging.warning(e)
                results_numeric.append(None)
        # Count the number of valid results
        valid_results = [r for r in results_numeric if r is not None]
        if len(valid_results) < len(results_numeric):
            logging.warning(
                f"Invalid results: {len(results_numeric) - len(valid_results)}"
            )
        return {"llmeval": float(np.mean(valid_results))}


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
        "llmeval": {"source": "custom", "average": True},
    }

    def __init__(self, metric_names: List[str]):
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
                elif metric_name == "llmeval":
                    self.metrics[metric_name] = LLMEval()

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
