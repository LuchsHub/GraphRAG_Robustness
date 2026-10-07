import time
import yaml

from neo4j import GraphDatabase

from paths import CONFIG_PATH
from .base import Retriever

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

VECTOR_SEARCH_CYPHER = config["retriever"]["vss"]["vector_search_cypher"]


class VSSRetriever(Retriever):
    def retrieve(self, query: str, top_k: int, entity_type: str) -> tuple[list, dict]:
        answer_ids = []
        log_dict = {}

        start_time = time.time()
        query_vector = self.ollama_embedder.embed_query(
            query,
            options={"temperature": self.temp, "seed": self.seed},
        )
        vector_search_cypher = VECTOR_SEARCH_CYPHER.format(
            label=f":{entity_type}" if entity_type else "",
            vector_index_name=self.vector_index_name,
            query_vector=query_vector,
            top_k=top_k,
        )
        records, _, _ = self.driver.execute_query(vector_search_cypher)
        log_dict["latency"] = time.time() - start_time

        log_dict["scores"] = {}
        for record in records:
            answer_id = record["id"]
            answer_ids.append(answer_id)
            log_dict["scores"][answer_id] = record["similarityScore"]

        return answer_ids, log_dict


driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j", "X"))
retriever = VSSRetriever(
    driver,
    ollama_embedder="qwen3-embedding:4b",
    vector_index_name="entity_index",
    temp=0.0,
    seed=7,
)

q = "Show me some throwing equipment from the brand DSP."

answer_ids, log_dict = retriever.retrieve(q, top_k=100, entity_type="product")

print(len(answer_ids), answer_ids[:15], log_dict)
