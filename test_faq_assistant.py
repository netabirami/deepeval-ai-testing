from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OllamaModel

from faq_assistant import get_faq_answer


def test_return_policy_question_is_relevant():
    question = "What is your return policy?"
    actual_output = get_faq_answer(question)

    model = OllamaModel(model="llama3.2:1b")

    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=model
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output
    )

    assert_test(test_case, [relevancy_metric])


def test_shipping_question_is_relevant():
    question = "How long does shipping take?"
    actual_output = get_faq_answer(question)

    model = OllamaModel(model="llama3.2:1b")

    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=model
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output
    )

    assert_test(test_case, [relevancy_metric])