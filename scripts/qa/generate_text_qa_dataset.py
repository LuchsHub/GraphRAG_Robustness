import yaml
import random
import csv
from typing import Literal

from pydantic import BaseModel
from stark_qa import load_skb
from neo4j import GraphDatabase
from ollama import generate


class QueryAmbiguityCheck(BaseModel):
    decision: Literal["KEEP", "DISCARD"]


CONFIG_FILE = "../../configs/config.yaml"
OUTPUT_FILE = "../../qa_datasets/text_amazon.csv"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SEED = config["models"]["seed"]
NEO4J_URI = config["neo4j"]["uri"]
NEO4J_USER = config["neo4j"]["user"]
NEO4J_PASSWORD = config["neo4j"]["password"]
NUM_QUERIES = config["textual_qa_dataset_generation"]["num_queries"]
QUERY_GEN_PROMPT = config["textual_qa_dataset_generation"]["query_gen_prompt"]
MODEL = config["models"]["qa_dataset_generation_model"]
TEMPERATURE = config["models"]["temperature"]
SIMILAR_NODES_CYPHER = config["textual_qa_dataset_generation"]["similar_nodes_cypher"]
AMBIGUITY_CHECK_PROMPT = config["textual_qa_dataset_generation"][
    "ambiguity_check_prompt"
]

skb = load_skb("amazon", download_processed=True)
random.seed(SEED)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

product_ids = skb.get_node_ids_by_type("product")
random_product_ids = random.sample(product_ids, k=NUM_QUERIES)

with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=["id", "query", "answer_ids"])
    writer.writeheader()

    written_queries = 0
    for pid in random_product_ids:
        doc = skb.get_doc_info(pid, add_rel=False)
        query_gen_prompt = QUERY_GEN_PROMPT.format(doc=doc)
        response = generate(
            model=MODEL,
            prompt=query_gen_prompt,
            options={"seed": SEED, "temperature": TEMPERATURE},
            think="high",
        )
        generated_query = response.response

        cypher_query = SIMILAR_NODES_CYPHER.format(id=pid)
        records, _, _ = driver.execute_query(
            cypher_query,
        )
        docs_string = f"-----Doc #0-----\n{doc}\n"
        for j, r in enumerate(records):
            docs_string += f"-----Doc #{j+1}-----\n"
            docs_string += r["document"] + "\n"

        query_ambiguity_prompt = AMBIGUITY_CHECK_PROMPT.format(
            query=generated_query, documents=docs_string
        )
        print(query_ambiguity_prompt)
        response = generate(
            model=MODEL,
            prompt=query_ambiguity_prompt,
            options={"temperature": TEMPERATURE, "seed": SEED},
            think="high",
            format=QueryAmbiguityCheck.model_json_schema(),
        )
        ambiguity_check = QueryAmbiguityCheck.model_validate_json(response.response)
        print(ambiguity_check)

        if ambiguity_check.decision == "KEEP":
            writer.writerow(
                {
                    "id": written_queries,
                    "query": generated_query,
                    "answer_ids": [pid],
                }
            )
            written_queries += 1
