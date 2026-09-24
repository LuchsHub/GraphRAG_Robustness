import yaml
import csv
import random

from neo4j import GraphDatabase
from ollama import generate

TEMPLATES_FILE = "../../configs/rel_templates.yaml"
CONFIG_FILE = "../../configs/config.yaml"
OUTPUT_FILE = "../../qa_datasets/rel_amazon.csv"

with open(TEMPLATES_FILE, "r", encoding="utf-8") as f:
    templates = yaml.safe_load(f)
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

LLM_SEED = config["models"]["seed"]
SELECTION_SEED = config["qa_dataset_generation"]["relational"]["seed"]
NEO4J_URI = config["neo4j"]["uri"]
NEO4J_USER = config["neo4j"]["user"]
NEO4J_PASSWORD = config["neo4j"]["password"]
QUERIES_PER_TEMPLATE = config["qa_dataset_generation"]["relational"][
    "queries_per_template"
]
PROMPT = config["qa_dataset_generation"]["relational"]["prompt"]
MODEL = config["models"]["qa_dataset_generation_model"]
TEMPERATURE = config["models"]["temperature"]


random.seed(SELECTION_SEED)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

row_id = 0
with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    writer = csv.DictWriter(
        outfile, fieldnames=["id", "template_id", "query", "answer_ids", "triples"]
    )
    writer.writeheader()

    for template in templates["templates"]:
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
                model=MODEL,
                prompt=prompt,
                options={"temperature": TEMPERATURE, "seed": LLM_SEED},
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
