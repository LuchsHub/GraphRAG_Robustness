from ollama import generate

PROMPT = """You are an intelligent assistant that generates queries about Amazon items.
I will provide you with a golden path from an Amazon product recommendation knowledge graph which leads to one product.
Your task is to create a natural-sounding customer query that leads to the target product as the answer.
Make sure to not confuse the product relations "also_view" and "also_buy" in the query.
Do not shorten product names in a way that could confuse them with similar products.

Path:
('chalk':category)-[:has_category]-(:product)-[:also_buy]-(target:product)-[:has_color]-('rainbow':color)

Query: """

response = generate(
    model="gemma4:26b",
    prompt=PROMPT,
    options={"temperature": 0.0},
    think=False,
)
print(
    "Thinking disabled----------------------------------------------------------------"
)
print(response.response)

for thinking_level in ["low", "medium", "high"]:
    response = generate(
        model="gemma4:26b",
        prompt=PROMPT,
        options={"temperature": 0.0},
        think=thinking_level,
    )
    print(
        f"Thinking level: {thinking_level}---------------------------------------------------------"
    )
    print(response.response)
