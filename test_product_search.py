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
    # Exact match: one right answer among near-misses (T006 right chest, T008 size S, T013 no print)
    ("white t-shirt, size L, with a print on the left side",
    "T002 - White Tee with Left Chest Print"),

    # Many correct answers: the AI must list all of them
    ("black t-shirt",
    "T004 - Black Graphic T-Shirt, T009 - Black Tee with Left Chest Print"),
    ("blue jeans size 32",
    "J001 - Slim Blue Jeans, J004 - Blue Jeans with Back Pocket Embroidery"),
    ("hoodie with a print on the back",
    "H004 - Navy Pullover Hoodie, H005 - Pink Hoodie with Back Print"),

    # Size trap: T002 is the same shirt in size L
    ("white t-shirt size S with a left chest print",
    "T008 - White Tee with Left Chest Print"),

    # Category trap: T002 is the same design as a t-shirt
    ("white hoodie with a left chest print, size L",
    "H003 - White Hoodie with Left Chest Print"),

    # Vague question: no color or size given
    ("tee with print on the left sleeve",
    "T003 - White Tee with Left Sleeve Print, T012 - Green Tee with Left Sleeve Print"),

    # No match: there is a red tee and other hoodies, but no red hoodie
    ("red hoodie",
    "No matches found."),
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
