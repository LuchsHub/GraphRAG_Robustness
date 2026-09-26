import time
import ast

from neo4j import Driver
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever


class VSSRetriever:
    def __init__(self, driver: Driver, index_name: str, ollama_model: str) -> None:
        self.ollama_embedder = OllamaEmbeddings(model=ollama_model)
        self.retriever = VectorRetriever(
            driver=driver,
            index_name=index_name,
            return_properties=["id"],
        )

    def retrieve(
        self, query: str, temperature: float, seed: int, top_k: int
    ) -> tuple[list, dict]:
        answer_ids = []
        log_dict = {}

        start_time = time.time()
        query_vector = self.ollama_embedder.embed_query(
            query,
            options={"temperature": temperature, "seed": seed},
        )
        search_results = self.retriever.search(
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
