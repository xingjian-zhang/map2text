import pytest
from llm4explore.utils.evaluate import Evaluation


class TestEvaluation:
    @pytest.fixture
    def predictions(self):
        return ["Hello World", "Goodbye World", "What is the meaning of life?"]

    @pytest.fixture
    def references(self):
        return ["Hey World", "Bye World", "What is the reason of life?"]

    @pytest.mark.parametrize(
        "metric_name",
        ["bleu", "rouge", "bertscore", "meteor", "bleurt", "cosine", "llmeval"],
    )
    def test_evaluate(self, metric_name, predictions, references):
        evaluation = Evaluation(metric_names=[metric_name])
        results = evaluation.compute(predictions, references)
        if metric_name == "bleu":
            assert len(results) > 0
        else:
            assert all([isinstance(value, (float, str)) for value in results.values()])
