from abc import ABC, abstractmethod

from neo4j import Driver
from neo4j_graphrag.embeddings import OllamaEmbeddings


class Retriever(ABC):
    def __init__(
        self, driver: Driver, ollama_embedder: str, temp: float, seed: int
    ) -> None:
        self.driver = driver
        self.ollama_embedder = OllamaEmbeddings(model=ollama_embedder)
        self.temp = temp
        self.seed = seed

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> tuple[list, dict]:
        pass
