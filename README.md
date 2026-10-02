# AI Product Search Testing (DeepEval + Ollama)

## What this project is
A small AI-powered product search feature for an e-commerce shop, tested using **DeepEval**, an open-source LLM testing framework.

Instead of using filters/dropdowns, a customer can type a natural-language request like:
> "white t-shirt, size L, with a print on the left side"

The AI reads the request and returns matching products from a sample inventory.

## Tech stack
- Python 3.11
- [DeepEval](https://github.com/confident-ai/deepeval) — LLM testing framework
- [Ollama](https://ollama.com) — runs the AI model locally (llama3.2:1b), free, no API key needed
- pytest (used internally by DeepEval)

## Project structure
```
products.json              # sample product inventory
product_search.py          # AI search feature (calls local LLM)
test_product_search.py     # DeepEval test suite
requirements.txt           # exact package versions used
```

## How to run this project

1. Install Python 3.10+ and [Ollama](https://ollama.com)
2. Pull the model: `ollama pull llama3.2:1b`
3. Start Ollama: `ollama serve`
4. Create a virtual environment and activate it:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
5. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
6. Run the test:
   ```
   deepeval test run test_product_search.py
   ```

## What I tested
Using DeepEval's `GEval` metric, I checked whether the AI's search results:
- Only include **white** products
- Only include products with a **left-side print** (chest or sleeve)
- Exclude black products and center/front prints

## Key finding
Running the same test 5 times in a row with a small local model (`llama3.2:1b`) as both the assistant and the judge resulted in a **0% pass rate**, even though manual spot-checks sometimes looked correct.

This highlights two real AI-testing challenges:
1. **Non-determinism** — the same AI input can produce different outputs each run, so a single manual check is not reliable evidence of quality.
2. **Judge-model reliability** — a small/weak model used as the evaluator can misjudge even correct answers, which affects trust in automated LLM evaluation pipelines.

## What I'd do next
- Compare pass rates using a larger judge model (e.g. Claude or GPT-4) vs. the small local model
- Add more test cases (different colors, sizes, print positions)
- Track pass rate over multiple runs instead of a single pass/fail



## Second feature tested: FAQ Assistant

A second AI feature that answers customer questions about shop policies (returns, shipping, payment), tested using DeepEval's **Answer Relevancy** metric — checking whether the AI's response actually addresses the question asked.

### Test results
| Test case | Score | Threshold | Result |
|---|---|---|---|
| "What is your return policy?" | 0.67 | 0.7 | FAILED |
| "How long does shipping take?" | 0.60 | 0.7 | FAILED |

### What I found
Both tests failed, just under the threshold. DeepEval's reasoning revealed two real quality issues:
- **Repetition** — the AI repeated the same statement, lowering relevancy
- **Vagueness** — the AI gave general information instead of directly answering the specific question (e.g. exact shipping time)

### Why this matters
This shows how automated relevancy testing can catch subtle AI quality issues (repetition, vagueness) that might be missed in casual manual review, especially with smaller/local models like `llama3.2:1b`.

## Final finding: judge-model reliability

After finding the FAQ Assistant tests failing (scores of 0.60–0.67, just under the 0.7 threshold), I improved the AI's prompt to be more direct and avoid repetition. The scores stayed the same (0.67) — but the judge model's own reasoning revealed the real issue:

> "The statement 'Returns are accepted within 30 days of purchase' is not relevant to addressing the question about return policy."

This statement is clearly relevant to a return policy question — meaning the **judge model itself made an incorrect judgment**, not that the AI's answer was actually bad.

### Conclusion
Using a small local model (`llama3.2:1b`) as both the assistant AND the judge introduces a reliability risk: the judge can misjudge correct answers. This is a well-known limitation of small LLMs used for evaluation, and a key reason production AI-testing pipelines often use a stronger, separate model (like GPT-4 or Claude) purely as the judge, even if a smaller/cheaper model powers the actual feature.

### Real-world takeaway
In a QA role, I would recommend:
- Using a stronger, independent model as the judge, separate from the model being tested
- Running each test multiple times and tracking pass-rate trends, not single pass/fail results
- Treating judge disagreement/contradictory reasoning as its own class of bug to report
_Last updated by hand from VS Code._
