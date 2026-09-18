from neo4j import GraphDatabase
from ollama import generate
import yaml
import csv
import random

SEED = 7
NEO4J_URI = "bolt://localhost:17687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "X"
TEMPLATES_FILE = "rel_templates.yaml"
QUERIES_PER_TEMPLATE = 1
PROMPT = """You are an intelligent assistant that generates queries about Amazon items.
I will provide you with a golden path from an Amazon product recommendation knowledge graph which leads to {num_answers} product(s).
Your task is to create a natural-sounding customer query that leads to the target product(s) as the answer.
Make sure to not confuse the product relations "also_view" and "also_buy" in the query.
Do not shorten product names in a way that may confuse them with similar products.

Path:
{path}

Query: """
OLLAMA_LLM = "gemma4:26b"
OUTPUT_FILE = "../qa_datasets/rel_amazon.csv"

random.seed(SEED)
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
        print(template["id"], template["name"])

        # get random assignments for current template
        records, _, _ = driver.execute_query(template["find_assignments_cypher"])
        sampled_assignments = random.sample(records, QUERIES_PER_TEMPLATE)

        for assignment in sampled_assignments:
            # instantiate assignment: get initial entity names, golden triples, answer count, answer ids
            records, _, _ = driver.execute_query(
                template["instantiate_assignment_cypher"], **assignment
            )
            instantiated_assignment = records[0]
            print(instantiated_assignment)

            llm_input = template["llm_input"].format(**instantiated_assignment)
            print(llm_input)

            prompt = PROMPT.format(
                path=llm_input, num_answers=instantiated_assignment["answer_count"]
            )

            response = generate(
                model=OLLAMA_LLM,
                prompt=prompt,
                options={"temperature": 0.0, "seed": SEED},
                think="high",
            )

            writer.writerow(
                {
                    "id": row_id,
                    "template_id": template["id"],
                    "query": response.response,
                    "answer_ids": instantiated_assignment["answer_ids"],
                    "triples": instantiated_assignment["triples"],
                }
            )
            print(response.response)

            row_id += 1

print("Fin.")
