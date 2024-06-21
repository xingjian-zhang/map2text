"""Entry point for quick start of the experiment."""

import argparse
import os
import re
import logging
import datetime

import yaml
import pandas as pd

from llm4explore.model.pretrain_map import PLMMapper

PATH_MATCHER = re.compile(r'\$\{([^}^{]+)\}')


def path_constructor(loader, node):
    """Fill ${ENV_VAR} in the configuration file."""
    value = node.value
    match = PATH_MATCHER.match(value)
    env_var = match.group()[2:-1]
    return os.environ.get(env_var) + value[match.end():]


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
        help=
        "Override the configuration. e.g. method.init_args.n_dims=3,data.path=data.csv",
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
    logging.basicConfig(level=logging.INFO,
                        filename=log_file,
                        format="%(asctime)s - %(message)s")
    logging.info(
        f"Configuration: {config}")  # NOTE: logging may expose api keys.

    # Run experiment.
    if config["type"] == "mapper":
        run_mapper_experiment(config)
    elif config["type"] == "generator":
        run_generator_experiment(config)
    else:
        raise ValueError(f"Unknown experiment type: {config['type']}.")


def run_mapper_experiment(config):
    # Load data.
    data = pd.read_csv(config["data"]["path"], sep="\t")
    targets = data[config["data"]["target_col"]].dropna().tolist()
    logging.info(f"Number of data points: {len(data)}")
    logging.info(f"Number of valid data points: {len(targets)}")

    # Run the mapper.
    if config["method"]["type"] == "plm_dr":
        mapper = PLMMapper(**config["method"]["init_args"])
    else:
        raise ValueError(f"Unknown method type: {config['method']['type']}.")
    mapper.encode_all(targets)


def run_generator_experiment(config):
    raise NotImplementedError("Generator experiment is not implemented.")


if __name__ == "__main__":
    main()
