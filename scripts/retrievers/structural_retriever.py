from typing import Literal
import time
import ast

from pydantic import BaseModel
from ollama import generate
from neo4j import Driver
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

PROMPT = """Given a list of entity types, identify all entities of those types from the query.

For each identified entity, extract the following information:
- name: Name of the entity.
- type: One of the following types: [product, brand, category, color]

Query: {query}

Output: """

class Entity(BaseModel):
    name: str 
    type: Literal["product", "brand", "category", "color"]

class Entities(BaseModel):
    entities: list[Entity]

QUERY = "I'm looking for a green product that people usually buy along with the Under Armour Men's HeatGear Armour Short Sleeve Compression Shirt."

query_gen_prompt = PROMPT.format(query=QUERY)
response = generate(
    model="gemma4:26b",
    prompt=query_gen_prompt,
    options={"seed": 7, "temperature": 0.0},
    think=False,
    format=Entities.model_json_schema(),
)

entities = Entities.model_validate_json(response.response)

print(entities)


""" class StructuralRetriever:
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
 """