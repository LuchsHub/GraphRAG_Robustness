from neo4j import GraphDatabase
from ollama import chat
import yaml
import csv

NEO4J_URI = "bolt://localhost:17687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "X"
TEMPLATES_FILE = "rel_templates.yaml"

SINGLE_TARGET_PROMPT = """You are an intelligent assistant that generates queries about Amazon items.
I will provide you with a golden path from an Amazon product recommendation knowledge graph which leads to one product.
Your task is to create a natural-sounding customer query that leads to the target product as the answer.
Do not shorten product names in a way that could confuse them with similar products.

Path:
{path}

Query: """

MULTI_TARGET_PROMPT = """You are an intelligent assistant that generates queries about Amazon items.
I will provide you with a golden path from an Amazon product recommendation knowledge graph which leads to multiple target products.
Your task is to create a natural-sounding customer query that leads to the target products as the answer.
Do not shorten product names in a way that could confuse them with similar products.

Path:
{path}

Query: """

OLLAMA_LLM = "gemma4:26b"
OUTPUT_FILE = "../qa_datasets/rel_amazon.csv"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

with open(TEMPLATES_FILE, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

row_id = 0
with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=["id", "hops", "query", "answer_ids"])
    writer.writeheader()

    for i, template in enumerate(config["templates"]):
        print(f"{template['name']} ({i+1}/{len(config['templates'])})")

        records, _, _ = driver.execute_query(template["sampling_cypher"])

        for j, record in enumerate(records):
            print(f"- {j + 1}/{len(records)}")
            llm_input = template["llm_input"].format(**record)

            if record["answer_count"] == 1:
                prompt = SINGLE_TARGET_PROMPT.format(path=llm_input)
            else:
                prompt = MULTI_TARGET_PROMPT.format(path=llm_input)

            response = chat(
                model=OLLAMA_LLM,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                options={"temperature": 0.0},
            )               # TODO: Retry on empty responses or timeouts
            writer.writerow(
                {
                    "id": row_id,
                    "hops": template["hops"],
                    "query": response.message.content,
                    "answer_ids": list(map(int, record["answer_ids"])),
                }
            )
            print(f"-- Path: {llm_input}\n-- Query: {response.message.content}")

            row_id += 1

print("Fin.")
