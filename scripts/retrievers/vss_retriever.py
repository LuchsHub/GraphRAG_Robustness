import ast
import time

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever

from .base import Retriever


class VSSRetriever(Retriever):
    def retrieve(self, query: str, top_k: int) -> tuple[list, dict]:
        answer_ids = []
        log_dict = {}

        start_time = time.time()
        query_vector = self.ollama_embedder.embed_query(
            query,
            options={"temperature": self.temp, "seed": self.seed},
        )
        retriever = VectorRetriever(
            driver=self.driver,
            index_name="product_index",
            return_properties=["id"],
        )
        search_results = retriever.search(
            query_vector=query_vector,
            top_k=top_k,
        )
        log_dict["latency"] = time.time() - start_time

        log_dict["scores"] = {}
        for result in search_results.items:
            content = ast.literal_eval(result.content)
            answer_id = content["id"]
            log_dict["scores"][answer_id] = result.metadata.get("score")
            answer_ids.append(answer_id)

        return answer_ids, log_dict


driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j", "X"))
retriever = VSSRetriever(driver, ollama_embedder="qwen3-embedding:4b", temp=0.0, seed=7)

q = "Show me some throwing equipment from the brand DSP."

answer_ids, log_dict = retriever.retrieve(q, top_k=100)

print(answer_ids[:15], log_dict)
