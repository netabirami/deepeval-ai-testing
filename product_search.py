import json
import re
import requests

# Common words that don't help tell products apart
STOP_WORDS = {"a", "an", "the", "with", "on", "in", "of", "for", "size", "print", "side", "color"}


def load_products():
    with open("products.json", "r") as f:
        return json.load(f)


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOP_WORDS}


def retrieve_products(user_query, top_k=3):
    """Retriever: score each product by how many query words it shares, keep the top_k."""
    query_words = tokenize(user_query)
    scored = []
    for product in load_products():
        product_words = tokenize(" ".join(str(v) for v in product.values()))
        score = len(query_words & product_words)
        if score > 0:
            scored.append((score, product))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [product for _, product in scored[:top_k]]


def search_products(user_query):
    """Returns {"answer": LLM's answer, "retrieval_context": retrieved products as strings}."""
    retrieved = retrieve_products(user_query)
    retrieval_context = [json.dumps(p) for p in retrieved]

    if not retrieved:
        return {"answer": "No matches found.", "retrieval_context": []}

    prompt = f"""You are a product search assistant for an e-commerce shop.
Here are the candidate products in JSON:
{chr(10).join(retrieval_context)}

Customer request: "{user_query}"

Return ONLY the products that match EVERY part of the request (color, size, print position).
List each match as "ID - Name". If nothing matches, say "No matches found."
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False
        }
    )

    return {"answer": response.json()["response"], "retrieval_context": retrieval_context}


if __name__ == "__main__":
    result = search_products("white t-shirt, size L, with a print on the left side")
    print("Retrieved:", result["retrieval_context"])
    print("Answer:", result["answer"])
