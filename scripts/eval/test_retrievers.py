import csv
import json
import os
import time
import yaml

from neo4j import GraphDatabase

from paths import ROOT_PATH, CONFIG_PATH
from scripts.retrievers import (
    VSSRetriever,
    StructuralRetriever,
    AgenticRetriever,
)

# configure which datasets to use
QA_DATASETS_FOLDER_PATH = ROOT_PATH / "qa_datasets"
QA_DATASETS = ["stark-amazon", "rel-amazon", "text-amazon"]
RESULTS_BASE_PATH = ROOT_PATH / "results" / "stark-amazon"
os.makedirs(RESULTS_BASE_PATH, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

NEO4J_URI = config["neo4j"]["uri"]
NEO4J_USER = config["neo4j"]["user"]
NEO4J_PASSWORD = config["neo4j"]["password"]
STANDARD_LABEL = config["graph"]["standard_label"]
VECTOR_INDEX_NAME_APPENDIX = config["graph"]["vector_index_name_appendix"]
FULLTEXT_INDEX_NAME_APPENDIX = config["graph"]["fulltext_index_name_appendix"]
LLM = config["models"]["retriever_llm"]
EMBEDDER = config["models"]["embedding_model"]
SEED = config["models"]["seed"]
TEMPERATURE = config["models"]["temperature"]
TOP_K = config["eval"]["top_k"]
ANSWER_TYPE = config["eval"]["answer_type"]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# configure which retrievers to evaluate
vss_retriever = VSSRetriever(
    driver=driver,
    ollama_embedder=EMBEDDER,
    vector_index_name_appendix=VECTOR_INDEX_NAME_APPENDIX,
    temp=TEMPERATURE,
    seed=SEED,
)
structural_retriever = StructuralRetriever(
    model=LLM,
    driver=driver,
    standard_label=STANDARD_LABEL,
    ollama_embedder=EMBEDDER,
    vector_index_name_appendix=VECTOR_INDEX_NAME_APPENDIX,
    temp=TEMPERATURE,
    seed=SEED,
)
agentic_retriever = AgenticRetriever(
    model=LLM,
    driver=driver,
    ollama_embedder=EMBEDDER,
    vector_index_name_appendix=VECTOR_INDEX_NAME_APPENDIX,
    standard_label=STANDARD_LABEL,
    temp=TEMPERATURE,
    seed=SEED,
)
retrievers = [vss_retriever, structural_retriever, agentic_retriever]

for dataset in QA_DATASETS:
    print(f"----- {dataset}")

    dataset_results_path = RESULTS_BASE_PATH / dataset
    os.makedirs(dataset_results_path, exist_ok=True)

    answers_writers = {}
    log_files = {}
    for retriever in retrievers:
        retriever_name = retriever.__class__.__name__

        answers_file_path = dataset_results_path / f"{retriever_name}.csv"
        answers_file = open(answers_file_path, "w", newline="", encoding="utf-8")

        writer = csv.writer(answers_file)
        writer.writerow(["query_id", "retrieved_ids"])
        answers_writers[retriever_name] = writer

        log_file_path = dataset_results_path / f"{retriever_name}_log.json"
        log_files[retriever_name] = open(log_file_path, "w", encoding="utf-8")

    with open(
        QA_DATASETS_FOLDER_PATH / f"{dataset}.csv", "r", encoding="utf-8"
    ) as dataset_file:
        reader = csv.DictReader(dataset_file)
        for row in reader:
            query = row["query"]
            query_id = row["id"]
            print(f"--- {query_id} {query}")

            for retriever in retrievers:
                retriever_name = retriever.__class__.__name__
                print(f"- {retriever_name}")
                start_time = time.time()
                retrieved_ids, log_dict = retriever.retrieve(
                    query, top_k=TOP_K, answer_entity_type=ANSWER_TYPE
                )
                log_dict["latency"] = round(time.time() - start_time, 3)
                print(retrieved_ids)

                answers_writers[retriever_name].writerow([query_id, retrieved_ids])

                log_dict["id"] = query_id
                json.dump(log_dict, log_files[retriever_name], indent=4)
