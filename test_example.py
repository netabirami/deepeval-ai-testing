from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ExactMatchMetric


def test_chatbot_says_correct_country():
    actual_output = "The capital of France is Paris"

    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output=actual_output,
        expected_output="The capital of France is Paris"
    )

    metric = ExactMatchMetric()
    assert_test(test_case, [metric])


