import csv
import random
import time
import yaml

from neo4j import GraphDatabase
import ollama

from paths import ROOT_PATH, CONFIG_PATH

TEMPLATES_FILE_PATH = ROOT_PATH / "configs" / "rel_templates.yaml"
OUTPUT_FILE_PATH = ROOT_PATH / "qa_datasets" / "rel_amazon.csv"
FAILED_GENERATIONS_FILE_PATH = ROOT_PATH / "qa_datasets" / "rel_amazon_timeouts.txt"

with open(TEMPLATES_FILE_PATH, "r", encoding="utf-8") as f:
    templates = yaml.safe_load(f)
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

LLM_SEED = config["models"]["seed"]
SELECTION_SEED = config["qa_dataset_generation"]["seed"]
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
client = ollama.Client(timeout=60.0)

row_id = 0
with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as outfile, open(
    FAILED_GENERATIONS_FILE_PATH, "w", encoding="utf-8"
) as fails_file:
    writer = csv.DictWriter(
        outfile, fieldnames=["id", "template_id", "query", "answer_ids", "triples"]
    )
    writer.writeheader()

    for template in templates["templates"]:
        print(template["id"], template["name"])

        # get random assignments for current template
        records, _, _ = driver.execute_query(template["find_assignments_cypher"])
        if len(records) <= QUERIES_PER_TEMPLATE:
            print(f"Not enough assignments. Using all {len(records)}.")
            assignments = records
        else:
            assignments = random.sample(records, QUERIES_PER_TEMPLATE)

        for assignment in assignments:
            print(row_id)

            # instantiate assignment: get initial entity names, golden triples, answer count, answer ids
            records, _, _ = driver.execute_query(
                template["instantiate_assignment_cypher"], **assignment
            )
            instantiated_assignment = records[0]

            llm_input = template["llm_input"].format(**instantiated_assignment)

            prompt = PROMPT.format(
                path=llm_input, num_answers=instantiated_assignment["answer_count"]
            )

            try:
                response = client.generate(
                    model=MODEL,
                    prompt=prompt,
                    options={"temperature": TEMPERATURE, "seed": LLM_SEED},
                    think="high",
                )
            except Exception as e:
                print(e)
                fails_file.write(f"{prompt}\n------------------------\n")
                time.sleep(5)
                continue

            writer.writerow(
                {
                    "id": row_id,
                    "template_id": template["id"],
                    "query": response.response,
                    "answer_ids": instantiated_assignment["answer_ids"],
                    "triples": instantiated_assignment["triples"],
                }
            )

            row_id += 1
