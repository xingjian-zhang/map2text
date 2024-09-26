"""Entry point for quick start of the experiment."""

import argparse
import datetime
import json
import logging
import os
import re
from typing import Any, List

import numpy as np
import pandas as pd
import yaml

from llm4explore.model.base import IdeaGenerator, IdeaMapper

PATH_MATCHER = re.compile(r"\$\{([^}^{]+)\}")


class MappingExperiment:
    def __init__(
        self,
        mapper: IdeaMapper,
        targets: List[str],
    ):
        self.mapper = mapper
        self.targets = targets

    @classmethod
    def from_config(cls, config: Any):
        from llm4explore.model import pretrain_map

        # Load data.
        data = pd.read_csv(config["data"]["path"], sep="\t")
        targets = data[config["data"]["target_col"]].dropna().tolist()
        logging.info(f"Number of data points: {len(data)}")
        logging.info(f"Number of valid data points: {len(targets)}")

        # Initialize the mapper.
        if config["method"]["type"] == "plm_dr":
            mapper = pretrain_map.PLMMapper(**config["method"]["init_args"])
        else:
            raise ValueError(f"Unknown method type: {config['method']['type']}.")
        return MappingExperiment(mapper, targets)

    def run(self):
        self.mapper.encode_all(self.targets)


class GenerationExperiment:
    def __init__(
        self,
        generator: IdeaGenerator,
        low_dim_embeddings_new: np.ndarray,
        targets_new: List[str],
        output_path: str,
        generator_type: str,
        num_tests: int = None,
    ):
        self.generator = generator
        self.low_dim_embeddings_new = low_dim_embeddings_new
        self.targets_new = targets_new
        self.output_path = output_path
        self.num_tests = num_tests or len(targets_new)
        self.generator_type = generator_type

    @classmethod
    def from_config(cls, config: Any):
        from llm4explore.model import non_trainable_gen, trainable_gen

        # Load text data.
        data = pd.read_csv(
            config["data"]["path"],
            sep="\t",
            usecols=[
                config["data"]["target_col"],
                config["data"]["time_col"],
                config["data"]["time_col"],
            ],
        )
        data = data.dropna(subset=[config["data"]["target_col"]])
        targets = data[config["data"]["target_col"]]
        logging.info(f"Number of data points: {len(data)}")
        logging.info(f"Number of valid data points: {len(targets)}")
        times = data[config["data"]["time_col"]]
        time_split = config["data"]["time_split"]

        # Load precomputed embeddings.
        npz = np.load(config["embeddings"]["path"])
        high_dim_embeddings = npz["high_dim_embeddings"]
        low_dim_embeddings = npz["low_dim_embeddings"]
        n_dims = low_dim_embeddings.shape[1]
        assert (
            len(targets) == high_dim_embeddings.shape[0]
        ), "Number of targets does not match the number of embeddings."

        # Split the data.
        targets_old = targets[times < time_split].tolist()
        targets_new = targets[times >= time_split].tolist()
        logging.info(f"Number of old data points: {len(targets_old)}")
        logging.info(f"Number of new data points: {len(targets_new)}")
        low_dim_embeddings_new = low_dim_embeddings[times >= time_split]
        low_dim_embeddings_old = low_dim_embeddings[times < time_split]

        # Initialize the generator.
        generator_type = config["method"]["type"]
        if generator_type == "plagiarism":
            generator = non_trainable_gen.PlagiarismGenerator(
                n_dims=n_dims,
                data_old=targets_old,
                low_dim_embeddings_old=low_dim_embeddings_old,
                **config["method"]["init_args"],
            )
        elif generator_type == "embedding":
            high_dim_embeddings_old = high_dim_embeddings[times < time_split]
            generator = non_trainable_gen.EmbeddingBasedGenerator(
                n_dims=n_dims,
                data_old=targets_old,
                low_dim_embeddings_old=low_dim_embeddings_old,
                high_dim_embeddings_old=high_dim_embeddings_old,
                **config["method"]["init_args"],
            )
        elif generator_type == "prompting":
            generator = non_trainable_gen.PromptingBasedGenerator(
                n_dims=n_dims,
                texts=targets_old,
                low_dim_embeddings=low_dim_embeddings_old,
                times=times[times < time_split].values,
                **config["method"]["init_args"],
            )
        elif generator_type == "finetune":
            generator = trainable_gen.FineTunedPLMGenerator(
                n_dims=n_dims,
                data_old=targets_old,
                low_dim_embeddings_old=low_dim_embeddings_old,
                **config["method"]["init_args"],
            )
        else:
            raise ValueError(f"Unknown generator type: {generator_type}.")
        return GenerationExperiment(
            generator,
            low_dim_embeddings_new,
            targets_new,
            config["output"],
            config["method"]["type"],
            config["data"]["num_tests"],
        )

    def run(self):
        queries = self.low_dim_embeddings_new[: self.num_tests]
        results = self.generator.decode_all(queries)
        if self.generator_type == "prompting":
            from llm4explore.utils.evaluate import Evaluation

            evaluation = Evaluation(metric_names=["cosine"])
            preds, logs, neighbors = zip(*results)
            targets = self.targets_new[: self.num_tests]
            outputs = []
            best_preds = []
            score_list = []
            for i,generate in enumerate(preds):
                data = json.loads(generate)
                generations = [each["key_idea"] for each in data['predictions']]
                references = neighbors[i]
                scores = []
                for gen in generations:
                    length = len(references)
                    gen = [gen]*length
                    score = evaluation.compute(gen, references)['cosine_cosine']
                    scores.append(score)
                score_gen_pairs = list(zip(scores, generations))
                sorted_score_gen_pairs = sorted(score_gen_pairs, key=lambda x: x[0], reverse=True)
                sorted_preds = [gen for _, gen in sorted_score_gen_pairs]
                sorted_scores = [score for score,_ in sorted_score_gen_pairs]
                score_list.append(sorted_scores)
                best_preds.append(sorted_preds)
            for i,pred in enumerate(preds):
                outputs.append(
                    {
                        "target": targets[i],
                        "prediction": best_preds[i],
                        "generations": pred,
                        "scores": score_list[i],
                        "queries": queries[i].tolist(),
                        "log": logs[i],
                    }
                )
            with open(self.output_path, "w") as f:
                json.dump(outputs, f, indent=4)
        else:
            preds, logs = zip(*results)
            targets = self.targets_new[: self.num_tests]
            outputs = []
            for i,pred in enumerate(preds):
                outputs.append(
                    {
                        "target": targets[i],
                        "prediction": pred,
                        "queries": queries[i].tolist(),
                        "log": logs[i],
                    }
                )
            with open(self.output_path, "w") as f:
                json.dump(outputs, f, indent=4)


def path_constructor(loader, node):
    """Fill ${ENV_VAR} in the configuration file."""
    value = node.value
    match = PATH_MATCHER.match(value)
    env_var = match.group()[2:-1]
    return os.environ.get(env_var) + value[match.end() :]


def main():
    # Load the configuration.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Override the configuration. e.g. method.init_args.n_dims=3,data.path=data.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output file.",
    )
    args = parser.parse_args()

    # Load environment variables.
    yaml.add_implicit_resolver("!path", PATH_MATCHER)
    yaml.add_constructor("!path", path_constructor)

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Override the configuration if necessary.
    if args.override:
        for override_arg in args.override.split(","):
            keys, value = override_arg.split("=")
            keys = keys.split(".")
            current = config
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value

    # Set up logging.
    config_basename_no_ext = os.path.splitext(os.path.basename(args.config))[0]
    log_dir = f"logs/{config_basename_no_ext}"
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{time_str}.log"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, filename=log_file, format="%(asctime)s - %(message)s"
    )
    logging.info(f"Configuration: {config}")  # NOTE: logging may expose api keys.

    # Run experiment.
    experiment_type = config["type"]
    if experiment_type == "mapping":
        MappingExperiment.from_config(config).run()
    elif experiment_type == "generation":
        GenerationExperiment.from_config(config).run()
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}.")


if __name__ == "__main__":
    main()
