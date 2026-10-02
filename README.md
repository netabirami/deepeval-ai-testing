# AI Testing with DeepEval + Ollama

Learning how to test AI features the way a QA engineer would — using **DeepEval**, an open-source LLM testing framework, and **free local models** via Ollama (no API keys, no cost).

The project is a small pretend online shop with AI features. Each feature is tested with a different group of DeepEval metrics. This is a learning-in-public project — I add a new feature and new findings as I go.

## Tech stack
- Python 3.11
- [DeepEval](https://github.com/confident-ai/deepeval) 4.2.2 — LLM testing framework
- [Ollama](https://ollama.com) — runs models locally, free
  - `llama3.2:1b` — the "production" AI being tested
  - `qwen2.5:3b` — the judge model that scores the AI
- pytest (used by DeepEval)
- Hardware: MacBook Air M1, 8 GB RAM

## Project structure
```
products.json              # sample product inventory
product_search.py          # AI product search (RAG: retriever + LLM)
test_product_search.py     # DeepEval tests for product search
faq_assistant.py           # AI FAQ assistant for shop policies
test_faq_assistant.py      # DeepEval tests for the FAQ assistant
test_example.py            # first practice test (ExactMatch)
test_capital.py            # first practice test (ExactMatch)
requirements.txt           # exact package versions used
```

## How to run

1. Install Python 3.10+ and [Ollama](https://ollama.com)
2. Pull the models:
   ```
   ollama pull llama3.2:1b
   ollama pull qwen2.5:3b
   ```
3. Start Ollama. On an 8 GB machine, keep only one model in memory at a time:
   ```
   OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 ollama serve
   ```
4. Create a virtual environment and install dependencies:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
5. Run the tests:
   ```
   deepeval test run test_product_search.py
   ```
   Run a single test case with `-k`, e.g. `deepeval test run test_product_search.py -k white`

---

## Feature 1: AI Product Search (RAG)

A customer types a natural-language request instead of using filters:
> "white t-shirt, size L, with a print on the left side"

### How it works
1. **Retriever** — a simple keyword retriever picks the top 3 candidate products from `products.json`
2. **Generator** — `llama3.2:1b` reads only those candidates and returns the matching products

`search_products()` returns both the answer **and** the retrieved products, so the tests can check each part separately.

### How it's tested
Each test case has a known correct answer (`expected_output`) and the products the retriever actually found (`retrieval_context`).

| Metric | What it checks | Part of the pipeline |
|---|---|---|
| GEval (Product Match Correctness) | Does the answer list exactly the expected products? | End-to-end |
| Faithfulness | Is every product in the answer actually in the retrieved products? (catches made-up products) | Generator |
| Contextual Recall | Did the retriever find the products needed for the correct answer? | Retriever |

Test cases:
| Query | Expected output |
|---|---|
| white t-shirt, size L, with a print on the left side | T002 – White Tee with Left Chest Print |
| black t-shirt | T004 – Black Graphic T-Shirt |
| red hoodie | No matches found. |

### Findings

**1. The original test had a gap.** The first version of the test only checked "white + left print". It missed **size L**, so a size M shirt (T003) would have passed. Adding an `expected_output` fixed this — comparing against a known correct answer is stricter than describing a rule.

**2. The AI was right, but the judge said it was wrong (false failure).** For the white t-shirt query, the AI answered correctly with only T002 — and even ignored two wrong products the retriever had passed it. But the judge scored it:

| Metric | Score | Was the judge right? |
|---|---|---|
| GEval Correctness | 0.2 ❌ | No — it penalised a friendly "I can help you with that" before the answer |
| Contextual Recall | 0.67 ❌ | No — T002 was retrieved, so recall should be 1.0 |
| Faithfulness | ERRORED ⚠️ | Timed out — no result |

**3. A bigger judge isn't always better — measure first.** I first tried `llama3.1:8b` as the judge. On 8 GB RAM it didn't fit in memory and ran at **0.3 tokens/second** (every judge call timed out). `qwen2.5:3b` runs fully on the GPU at **33 tokens/second**. The right model is the one that actually works on your hardware.

**4. ERRORED ≠ FAILED.** A metric that times out hasn't checked anything. It's an infrastructure problem, not a quality result — reporting it as either "the AI is fine" or "the AI is broken" would be wrong.

### Earlier finding (1b judge)
Before the RAG upgrade, I used `llama3.2:1b` as both the AI and the judge. Running the same test 5 times gave a **0% pass rate**, even though manual spot-checks sometimes looked correct. This showed two real AI-testing challenges:
1. **Non-determinism** — the same input can produce different outputs each run, so one manual check isn't reliable evidence.
2. **Judge reliability** — a small judge can misjudge correct answers.

---

## Feature 2: FAQ Assistant

Answers customer questions about shop policies (returns, shipping, payment), tested with DeepEval's **Answer Relevancy** metric (judge: `llama3.2:1b`).

| Test case | Score | Threshold | Result |
|---|---|---|---|
| "What is your return policy?" | 0.67 | 0.7 | FAILED |
| "How long does shipping take?" | 0.60 | 0.7 | FAILED |

### Findings
- **Repetition** and **vagueness** in the AI's answers lowered relevancy.
- After improving the prompt, the score stayed at 0.67 — and the judge's own reasoning showed the real issue:
  > "The statement 'Returns are accepted within 30 days of purchase' is not relevant to addressing the question about return policy."

  That statement is clearly relevant, so **the judge itself was wrong**.

---

## Key takeaways so far
- A failed test isn't always a bug in the AI — sometimes it's a bug in the judge. Read the judge's reasoning before reporting.
- Use a judge model that is separate from (and ideally stronger than) the model being tested.
- Compare against a known correct answer (`expected_output`) where possible, not just a written rule.
- Separate infrastructure errors (timeouts) from quality failures.
- Run tests multiple times and track pass rates, not single pass/fail results.

## Roadmap
- [x] Product search as a RAG pipeline with `expected_output` and `retrieval_context`
- [ ] Feature 1: bigger catalog, all 5 RAG metrics, fix the Faithfulness timeout, use GEval `evaluation_steps`
- [ ] Feature 2: FAQ Assistant — Hallucination (with `context`), Answer Relevancy
- [ ] Feature 3: Review Summarizer — Summarization
- [ ] Feature 4: Order Extractor — JSON Correctness, Pattern Match
- [ ] Feature 5: Order Agent with tools — Tool Correctness, Argument Correctness, Task Completion
- [ ] Feature 6: Support Chatbot (multi-turn) — Knowledge Retention, Conversation Completeness, Role Adherence, Turn Relevancy
- [ ] Feature 7: Safety — Bias, Toxicity, PII Leakage, Misuse, Role Violation
- [ ] Feature 8: Custom metrics — GEval, DAG
