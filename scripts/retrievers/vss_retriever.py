import time

from configs.retriever_configs import VSSRetrieverConfig
from .base import Retriever


class VSSRetriever(Retriever):
    def retrieve(
        self, query: str, top_k: int, answer_entity_type: str
    ) -> tuple[list, dict]:
        answer_ids = []
        log_dict = {}

        start_time = time.time()
        query_vector = self.ollama_embedder.embed_query(
            query,
            options={"temperature": self.temp, "seed": self.seed},
        )
        vector_index_name = f"{answer_entity_type}{self.vector_index_name_appendix}"
        vector_search_cypher = VSSRetrieverConfig.vector_search_cypher.format(
            label=answer_entity_type,
            vector_index_name=vector_index_name,
            query_vector=query_vector,
            top_k=top_k,
        )
        records, _, _ = self.driver.execute_query(vector_search_cypher)
        latency = time.time() - start_time
        log_dict["latency"] = round(latency, 3)

        log_dict["scores"] = {}
        for record in records:
            answer_id = record["id"]
            answer_ids.append(answer_id)
            log_dict["scores"][answer_id] = round(record["similarityScore"], 3)

        return answer_ids, log_dict
