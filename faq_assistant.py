import requests

def get_faq_answer(customer_question):
    shop_policies = """
Return Policy: Items can be returned within 30 days of purchase, unused and with tags attached.
Shipping: Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.
Payment: We accept credit cards, PayPal, and Apple Pay. We do not accept cash on delivery.
"""

    prompt = f"""You are a customer support assistant for an e-commerce shop.
Only use the information below to answer the customer's question. Do not make up information.

Shop policies:
{shop_policies}

Customer question: "{customer_question}"

Instructions:
- Answer in ONE short sentence
- Directly state the specific number or fact requested (e.g. exact number of days)
- Do NOT repeat yourself
- Do NOT add unrelated information
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


if __name__ == "__main__":
    answer = get_faq_answer("Within how many days can I return an item?")
    print(answer)
