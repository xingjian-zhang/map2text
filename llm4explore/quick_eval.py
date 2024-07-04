"""Entry point for quick evaluation of the experiment."""

import argparse
import json

import yaml

from llm4explore.utils.evaluate import Evaluation


def safe_load_answer_from_json(raw_str: str):
    """Safely load the answer from the JSON string."""
    try:
        predict = json.loads(raw_str)
        answer = predict["predictions"][0]['key_idea']
    except json.JSONDecodeError:
        return None
    if isinstance(answer, str) and len(answer) > 0:
        return answer


def main():
    # Process the arguments.
    parser = argparse.ArgumentParser(description="Quick evaluation of the experiment.")
    parser.add_argument(
        "experiment_file",
        type=str,
        help="Path to the experiment output file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the evaluation metric output file.",
        default=None,
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="The evaluation metric. Specify multiple metrics separated by comma."
        "e.g. --metrics=bleu,rouge",
        default="all",
    )
    parser.add_argument(
        "--process_json",
        action="store_true",
        help="Whether the experiment output is in the JSON format, instructed by prompt.",
    )
    args = parser.parse_args()
    experiment_file = args.experiment_file
    output_file = args.output_file
    metrics = args.metrics
    if metrics == "all":
        metrics = None
    else:
        metrics = metrics.split(",")
    if output_file is None:
        output_file = experiment_file.replace(".json", "_evaluation.yaml")

    # Load the experiment output.
    with open(experiment_file, "r") as file:
        experiment_output = json.load(file)
        print(f"Loaded {len(experiment_output)} outputs from {experiment_file}.")
        predictions, references = [], []
        for output in experiment_output:
            prediction = output["prediction"]
            reference = output["target"]
            if args.process_json:
                prediction = safe_load_answer_from_json(prediction)
                if prediction is None:
                    continue
            predictions.append(prediction)
            references.append(reference)
        print(f"Processed {len(predictions)} valid outputs.")

    # Evaluate the experiment.
    evaluation = Evaluation(metric_names=metrics)
    results = evaluation.compute(predictions, references)
    print("Evaluation results:")
    print(yaml.dump(results, indent=4))

    # Save the evaluation results.
    with open(output_file, "w") as file:
        yaml.dump(results, file, indent=4)


if __name__ == "__main__":
    main()
