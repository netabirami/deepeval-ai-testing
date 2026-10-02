from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ExactMatchMetric


def test_france_capital():
    question = "What is the capital of France?"

    # Pretend this came from your AI/chatbot
    actual_output = "The capital of France is Paris"

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output="The capital of France is Paris"
    )

    metric = ExactMatchMetric()
    assert_test(test_case, [metric])