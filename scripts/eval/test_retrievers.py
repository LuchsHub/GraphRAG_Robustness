import csv
import yaml

from neo4j import GraphDatabase

from paths import ROOT_PATH, CONFIG_PATH
from scripts.retrievers import (
    VSSRetriever,
    StructuralRetriever,
    AgenticRetriever,
)

QA_DATASETS_FOLDER_PATH = ROOT_PATH / "qa_datasets"
QA_DATASETS = ["rel-amazon", "stark-amazon", "text-amazon"]
RESULTS_FOLDER_PATH = ROOT_PATH / "results" / "stark-amazon"

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
    fulltext_index_name_appendix=FULLTEXT_INDEX_NAME_APPENDIX,
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
    print(f"-----{dataset}-----")
    with open(QA_DATASETS_FOLDER_PATH / f"{dataset}.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row["query"]
            answer_ids = row["answer_ids"]
            print(f"---{query}---")
            print(answer_ids)
            for retriever in retrievers:
                print(f"-{retriever.__class__.__name__}-")
                retrieved_ids, log_dict = retriever.retrieve(query, top_k=TOP_K, answer_entity_type=ANSWER_TYPE)
                print(retrieved_ids)
