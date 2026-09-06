from neo4j import GraphDatabase
from ollama import generate
import yaml
import csv

NEO4J_URI = "bolt://localhost:17687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "X"
TEMPLATES_FILE = "rel_templates.yaml"

SINGLE_TARGET_PROMPT = """You are an intelligent assistant that generates queries about Amazon items.
I will provide you with a golden path from an Amazon product recommendation knowledge graph which leads to one product.
Your task is to create a natural-sounding customer query that leads to the target product as the answer.
Make sure to not confuse the product relations "also_view" and "also_buy" in the query.
Do not shorten product names in a way that could confuse them with similar products.

Path:
{path}

Query: """

MULTI_TARGET_PROMPT = """You are an intelligent assistant that generates queries about Amazon items.
I will provide you with a golden path from an Amazon product recommendation knowledge graph which leads to multiple target products.
Your task is to create a natural-sounding customer query that leads to the target products as the answer.
Make sure to not confuse the product relations "also_view" and "also_buy" in the query.
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
    writer = csv.DictWriter(
        outfile, fieldnames=["id", "template_id", "query", "answer_ids", "triples"]
    )
    writer.writeheader()

    for template in config["templates"]:
        print(f"({template['id']}) {template['name']}")

        records, _, _ = driver.execute_query(template["sampling_cypher"])
        num_records = len(records)

        for i, record in enumerate(records):
            print(f"- {i + 1}/{num_records}")
            llm_input = template["llm_input"].format(**record)
            print(f"-- Path: {llm_input}")

            if record["answer_count"] == 1:
                prompt = SINGLE_TARGET_PROMPT.format(path=llm_input)
            else:
                prompt = MULTI_TARGET_PROMPT.format(path=llm_input)

            response = generate(
                model=OLLAMA_LLM,
                prompt=prompt,
                options={"temperature": 0.0, "seed": 7},
                think="low",
            )

            triples = []
            for path in record["paths"]:
                for rel in path.relationships:
                    h = rel.start_node.get("id")
                    t = rel.end_node.get("id")
                    r = rel.type

                    if (h, r, t) not in triples and (t, r, h) not in triples:
                        triples.append((h, r, t))

            writer.writerow(
                {
                    "id": row_id,
                    "template_id": template["id"],
                    "query": response.response,
                    "answer_ids": list(map(int, record["answer_ids"])),
                    "triples": triples,
                }
            )
            print(f"-- Query: {response.response}")

            row_id += 1

print("Fin.")
