from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from peft import AutoPeftModelForSeq2SeqLM

from llm4explore.model.base import IdeaGenerator
from llm4explore.model.common import ANNSampler
from llm4explore.model.trainable_gen.data import make_input


class FineTunedPLMGenerator(IdeaGenerator):
    """Fine-tuned PLM-based idea generator."""

    def __init__(
        self,
        n_dims: int,
        data_old: List[str],
        low_dim_embeddings_old: np.ndarray,
        checkpoint_dir: str,
        use_sampler: bool = False,
        sampler_kwargs: Dict[str, Any] = None,
        input_kwargs: Dict[str, Any] = None,
        batch_size: int = 1,
    ):
        super().__init__(n_dims, data_old, low_dim_embeddings_old)
        sampler_kwargs = sampler_kwargs or {}
        input_kwargs = input_kwargs or {}
        self.input_kwargs = input_kwargs
        if use_sampler:
            self.sampler = ANNSampler(low_dim_embeddings_old, **sampler_kwargs)
        else:
            self.sampler = None
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir

        # load model and tokenizer
        self.model = AutoPeftModelForSeq2SeqLM.from_pretrained(checkpoint_dir, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.pipe = pipeline(
            task="text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            framework="pt",
            do_sample=False,
            max_length=64,
            repetition_penalty=1.0,
        )

    def decode(self, low_dim_embedding: np.ndarray) -> Tuple[str, Any]:
        raise NotImplementedError(
            "Fine-tuned PLM generator does not support single decoding."
        )

    def decode_all(self, low_dim_embeddings: np.ndarray) -> List[Tuple[str, Any]]:
        input_texts = []
        for query in low_dim_embeddings:
            if self.sampler is not None:
                indices, dists = self.sampler.sample(query)
                reference_texts = [self.data_old[i] for i in indices]
                reference_embeddings = self.low_dim_embeddings_old[indices]
            else:
                reference_texts = None
                reference_embeddings = None
            input_texts.append(
                make_input(
                    query,
                    reference_embeddings=reference_embeddings,
                    reference_texts=reference_texts,
                    **self.input_kwargs,
                )
            )
        preds = []
        for out in self.pipe(input_texts, batch_size=self.batch_size):
            preds.append(out["generated_text"])
        return list(zip(preds, input_texts))
