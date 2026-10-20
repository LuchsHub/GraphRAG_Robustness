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

    answers_files = {}
    answers_writers = {}
    completed_ids = {}
    for retriever in retrievers:
        retriever_name = retriever.__class__.__name__

        answers_file_path = dataset_results_path / f"{retriever_name}.csv"

        completed_ids[retriever_name] = set()
        if os.path.exists(answers_file_path):
            with open(answers_file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for line in reader:
                    if line:
                        completed_ids[retriever_name].add(int(line[0]))
        
        answers_file = open(answers_file_path, "a", newline="", encoding="utf-8")
        writer = csv.writer(answers_file)

        file_already_existed = (
            os.path.exists(answers_file_path) and os.path.getsize(answers_file_path) > 0
        )
        if not file_already_existed:
            writer.writerow(["query_id", "retrieved_ids"])
            answers_file.flush()

        answers_files[retriever_name] = answers_file
        answers_writers[retriever_name] = writer

        os.makedirs(dataset_results_path / f"{retriever_name}_logs", exist_ok=True)

    with open(
        QA_DATASETS_FOLDER_PATH / f"{dataset}.csv", "r", encoding="utf-8"
    ) as dataset_file:
        reader = csv.DictReader(dataset_file)
        for row in reader:
            query = row["query"]
            query_id = int(row["id"])
            print(f"--- {query_id} {query}")

            for retriever in retrievers:
                retriever_name = retriever.__class__.__name__

                if query_id in completed_ids[retriever_name]:
                    continue

                print(f"- {retriever_name}")
                start_time = time.time()
                retrieved_ids, log_dict = retriever.retrieve(
                    query, top_k=TOP_K, answer_entity_type=ANSWER_TYPE
                )
                log_dict["latency"] = round(time.time() - start_time, 3)
                log_dict["id"] = query_id
                print(retrieved_ids)

                answers_writers[retriever_name].writerow([query_id, retrieved_ids])
                answers_files[retriever_name].flush()

                log_path = (
                    dataset_results_path / f"{retriever_name}_logs" / f"{query_id}.json"
                )
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(log_dict, f, indent=4)

                completed_ids[retriever_name].add(query_id)
