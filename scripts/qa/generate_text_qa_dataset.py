import csv
import random
import time
import yaml
from typing import Literal

import ollama
from neo4j import GraphDatabase
from pydantic import BaseModel
from stark_qa import load_skb

from paths import CONFIG_PATH, ROOT_PATH


class QueryAmbiguityCheck(BaseModel):
    decision: Literal["KEEP", "DISCARD"]


OUTPUT_FILE_PATH = ROOT_PATH / "qa_datasets" / "text_amazon.csv"
FAILED_GENERATIONS_FILE_PATH = ROOT_PATH / "qa_datasets" / "text_amazon_timeouts.txt"
DISCARDED_QUERIES_FILE_PATH = ROOT_PATH / "qa_datasets" / "text_amazon_discarded.txt"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

LLM_SEED = config["models"]["seed"]
SELECTION_SEED = config["qa_dataset_generation"]["seed"]
NEO4J_URI = config["neo4j"]["uri"]
NEO4J_USER = config["neo4j"]["user"]
NEO4J_PASSWORD = config["neo4j"]["password"]
NUM_QUERIES = config["qa_dataset_generation"]["textual"]["num_queries"]
QUERY_GEN_PROMPT = config["qa_dataset_generation"]["textual"]["query_gen_prompt"]
MODEL = config["models"]["qa_dataset_generation_model"]
TEMPERATURE = config["models"]["temperature"]
SIMILAR_NODES_CYPHER = config["qa_dataset_generation"]["textual"][
    "similar_nodes_cypher"
]
AMBIGUITY_CHECK_PROMPT = config["qa_dataset_generation"]["textual"][
    "ambiguity_check_prompt"
]

skb = load_skb("amazon", download_processed=True)
random.seed(SELECTION_SEED)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
client = ollama.Client(timeout=60.0)

# get random products
product_ids = skb.get_node_ids_by_type("product")
random_product_ids = random.sample(product_ids, k=NUM_QUERIES)

with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as outfile, open(
    FAILED_GENERATIONS_FILE_PATH, "w", encoding="utf-8"
) as fails_file, open(
    DISCARDED_QUERIES_FILE_PATH, "w", encoding="utf-8"
) as discarded_file:
    writer = csv.DictWriter(outfile, fieldnames=["id", "query", "answer_ids"])
    writer.writeheader()

    written_queries = 0
    for pid in random_product_ids:
        doc = skb.get_doc_info(pid, add_rel=False)

        # generate query based on document
        query_gen_prompt = QUERY_GEN_PROMPT.format(doc=doc)
        try:
            response = client.generate(
                model=MODEL,
                prompt=query_gen_prompt,
                options={"seed": LLM_SEED, "temperature": TEMPERATURE},
                think="high",
            )
        except Exception as e:
            print(e)
            fails_file.write(f"{query_gen_prompt}\n------------------------\n")
            time.sleep(5)
            continue
        generated_query = response.response

        # get docs similar to this product's doc and concatenate them into one string
        cypher_query = SIMILAR_NODES_CYPHER.format(id=pid)
        records, _, _ = driver.execute_query(
            cypher_query,
        )
        docs_string = f"-----Doc #0-----\n{doc}\n"
        for j, r in enumerate(records):
            docs_string += f"-----Doc #{j+1}-----\n"
            docs_string += r["document"] + "\n"

        # check the generated query for ambiguity (can it be answered by similar docs as well?)
        query_ambiguity_prompt = AMBIGUITY_CHECK_PROMPT.format(
            query=generated_query, documents=docs_string
        )
        try:
            response = client.generate(
                model=MODEL,
                prompt=query_ambiguity_prompt,
                options={"temperature": TEMPERATURE, "seed": LLM_SEED},
                think="high",
                format=QueryAmbiguityCheck.model_json_schema(),
            )
        except Exception as e:
            print(e)
            fails_file.write(f"{query_ambiguity_prompt}\n------------------------\n")
            time.sleep(5)
            continue
        ambiguity_check = QueryAmbiguityCheck.model_validate_json(response.response)
        if ambiguity_check.decision == "KEEP":
            writer.writerow(
                {
                    "id": written_queries,
                    "query": generated_query,
                    "answer_ids": [pid],
                }
            )
            written_queries += 1
        else:
            discarded_file.write(
                f"{query_ambiguity_prompt}\n------------------------\n"
            )
