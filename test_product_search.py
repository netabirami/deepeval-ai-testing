import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRecallMetric
from deepeval.models import OllamaModel

from product_search import search_products

# Judge is a stronger, separate model from the one powering the app (llama3.2:1b)
judge = OllamaModel(model="qwen2.5:3b", temperature=0)

# (customer query, expected_output = the correct answer we know from products.json)
TEST_CASES = [
    ("white t-shirt, size L, with a print on the left side", 
    "T002 - White Tee with Left Chest Print"),
    ("black t-shirt", "T004 - Black Graphic T-Shirt"),
    ("red hoodie", "No matches found."),
]


@pytest.mark.parametrize("query, expected_output", TEST_CASES)
def test_product_search(query, expected_output):
    result = search_products(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=result["answer"],
        expected_output=expected_output,
        retrieval_context=result["retrieval_context"],
    )

    # End-to-end: does the answer list exactly the same products as the expected answer?
    correctness_metric = GEval(
        name="Product Match Correctness",
        criteria="Check that the actual output lists exactly the same product IDs as the expected output, with no missing and no extra products. If the expected output is 'No matches found.', the actual output must also say no products match. Ignore differences in wording or formatting.",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=judge,
        threshold=0.7,
        async_mode=False
    )
    metrics = [correctness_metric]

    # RAG metrics need retrieved products to judge against
    if result["retrieval_context"]:
        # Generator: is every product in the answer actually in the retrieved products?
        metrics.append(FaithfulnessMetric(threshold=0.7, model=judge, async_mode=False))
        # Retriever: did retrieval find the products needed for the expected answer?
        metrics.append(ContextualRecallMetric(threshold=0.7, model=judge, async_mode=False))

    assert_test(test_case, metrics)
