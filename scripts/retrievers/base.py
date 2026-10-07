from abc import ABC, abstractmethod

from neo4j import Driver
from neo4j_graphrag.embeddings import OllamaEmbeddings


class Retriever(ABC):
    def __init__(
        self,
        driver: Driver,
        ollama_embedder: str,
        vector_index_name: str,
        temp: float,
        seed: int,
    ) -> None:
        self.driver = driver
        self.ollama_embedder = OllamaEmbeddings(model=ollama_embedder)
        self.vector_index_name = vector_index_name
        self.temp = temp
        self.seed = seed

    @abstractmethod
    def retrieve(self, query: str, top_k: int, entity_type: str) -> tuple[list, dict]:
        pass
