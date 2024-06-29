import argparse
import datetime
import logging
import os

import numpy as np
import pandas as pd
import transformers
import yaml

from llm4explore.model.trainable_gen import data
from llm4explore.utils.evaluate import Evaluation


def main():
    # Load the configuration,
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

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

    training_args = transformers.Seq2SeqTrainingArguments(**config["training_args"])
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(config["model"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(config["tokenizer"])

    # Load the dataset.
    raw_data = pd.read_csv(
        config["data"]["path"],
        sep="\t",
        usecols=[
            config["data"]["target_col"],
            config["data"]["time_col"],
            config["data"]["time_col"],
        ],
    )
    raw_data = raw_data.dropna(subset=[config["data"]["target_col"]])
    targets = raw_data[config["data"]["target_col"]]
    times = raw_data[config["data"]["time_col"]]
    logging.info(f"Number of data points: {len(raw_data)}")
    logging.info(f"Number of valid data points: {len(targets)}")

    # Load precomputed embeddings.
    npz = np.load(config["embeddings"]["path"])
    low_dim_embeddings = npz["low_dim_embeddings"]

    ds = data.get_dataset(
        texts=targets,
        times=times,
        low_dim_embeddings=low_dim_embeddings,
        time_train=config["data"]["time_train"],
        time_val=config["data"]["time_val"],
        time_test=config["data"]["time_test"],
        use_sampler=config["data"]["use_sampler"],
        sampler_kwargs=config["data"]["sampler_kwargs"],
        input_kwargs=config["data"]["input_kwargs"],
    )

    evaluation = Evaluation(metric_names=config["metrics"])

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return evaluation.compute(decoded_preds, decoded_labels)

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer),
        train_dataset=ds["train"].with_format("torch"),
        eval_dataset=ds["test"].with_format("torch"),
        compute_metrics=compute_metrics,
    )

    trainer.train()
