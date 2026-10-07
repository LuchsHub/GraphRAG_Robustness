import ast
import time
import yaml

from ollama import generate
from pydantic import BaseModel
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever

from paths import CONFIG_PATH
from .base import Retriever

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

NER_PROMPT = config["retriever"]["structural"]["NER_prompt"]
TEXT_SEARCH_CYPHER = config["retriever"]["structural"]["text_search_cypher"]
HOP_SCORING = config["retriever"]["structural"]["hop_scoring"]
H_HOP_NEIGHBORS_CYPHER = config["retriever"]["structural"]["h_hop_neighbors_cypher"]
VECTOR_RANK_CYPHER = config["retriever"]["structural"]["vector_rank_cypher"]


class Entities(BaseModel):
    entities: list[str]


class StructuralRetriever(Retriever):
    def __init__(self, model: str, fulltext_index_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.fulltext_index_name = fulltext_index_name

    def retrieve(self, query: str, top_k: int, entity_type: str) -> tuple[list, dict]:
        answer_ids = []
        log_dict = {}

        start_time = time.time()

        # NER with a shortened version of the PathRAG NER prompt
        mentioned_entities = self.extract_entities_from_query(query)

        # link every entity to KG
        seeds = []
        log_dict["seed_entities"] = []
        for mention in mentioned_entities.entities:
            id, used_vss = self.link_entity_to_graph(mention)
            if id not in seeds:
                seeds.append(id)
            log_dict["seed_entities"].append(
                {
                    "mention": mention,
                    "linked_id": id,
                    "used_vss": used_vss,
                }
            )

        # vectorize query
        query_emb = self.ollama_embedder.embed_query(
            query,
            options={"temperature": self.temp, "seed": self.seed},
        )

        # treat query as an entity, similar to KAR
        query_entity_id = self.find_query_entity(query_emb)
        log_dict["query_entity_id"] = query_entity_id
        if query_entity_id not in seeds:
            seeds.append(query_entity_id)

        # score and group by structural score
        scores_by_id = self.compute_structural_scores(seeds)

        # break ties with vector similarity, return top_k
        ranked_entities = self.rank_by_vector_similarity(
            scores_by_id, query_emb, top_k, entity_type
        )
        answer_ids = [entity["id"] for entity in ranked_entities]
        log_dict["scores"] = ranked_entities

        log_dict["latency"] = time.time() - start_time

        return answer_ids, log_dict

    def extract_entities_from_query(self, query: str) -> Entities:
        """Extract mentioned entities from the user query."""
        query_gen_prompt = NER_PROMPT.format(query=query)
        response = generate(
            model=self.model,
            prompt=query_gen_prompt,
            options={"seed": self.seed, "temperature": self.temp},
            think=False,
            format=Entities.model_json_schema(),
        )

        return Entities.model_validate_json(response.response)

    def link_entity_to_graph(self, mention: str) -> tuple[str, bool]:
        """Link an entity to the knowledge graph through exact-match name search, alternatively through document-based VSS."""

        # try case-insensitive exact-matching
        text_search_cypher = TEXT_SEARCH_CYPHER.format(
            index_name=self.fulltext_index_name, entity_name=mention.lower()
        )
        records, _, _ = self.driver.execute_query(text_search_cypher)
        if records:
            return records[0]["id"], False

        # resort to VSS
        else:
            mention_vector = self.ollama_embedder.embed_query(
                mention,
                options={"temperature": self.temp, "seed": self.seed},
            )
            retriever = VectorRetriever(
                driver=self.driver,
                index_name=self.vector_index_name,
                return_properties=["id"],
            )
            results = retriever.search(
                query_vector=mention_vector,
                top_k=5,
            )
            result = results.items[0]
            content = ast.literal_eval(result.content)
            return content["id"], True

    def find_query_entity(self, query_emb: list[float]) -> str:
        """Find a KG entity based on the query embedding."""
        retriever = VectorRetriever(
            driver=self.driver,
            index_name=self.vector_index_name,
            return_properties=["id"],
        )
        results = retriever.search(
            query_vector=query_emb,
            top_k=5,
        )
        result = results.items[0]
        content = ast.literal_eval(result.content)

        return content["id"]

    def compute_structural_scores(self, seeds: list[str]) -> dict[float, list[str]]:
        """Get KG entities based on their distance to seed entities."""
        # set score = 1 for product seeds, since they themselves might be answers
        scores = {id: HOP_SCORING[0] for id in seeds}

        for id in seeds:
            visited = set()
            visited.add(id)

            for hop in range(1, len(HOP_SCORING)):
                # get all exactly-h-hop neighbors
                neighbors_cypher_query = H_HOP_NEIGHBORS_CYPHER.format(id=id, h=hop)
                records, _, _ = self.driver.execute_query(neighbors_cypher_query)

                # add score based on HOP_SCORING
                for neighbor in records[0]["ids"]:
                    if neighbor not in visited:
                        scores[neighbor] = scores.get(neighbor, 0) + HOP_SCORING[hop]
                        visited.add(neighbor)

        # group by structural score for pruning
        ids_by_score = {}
        for id, score in scores.items():
            if score not in ids_by_score:
                ids_by_score[score] = []
            ids_by_score[score].append(id)
        ids_by_score = dict(sorted(ids_by_score.items(), reverse=True))

        return ids_by_score

    def rank_by_vector_similarity(
        self,
        ids_by_score: dict[float, list[str]],
        query_emb: list[float],
        top_k: int,
        entity_type: str,
    ) -> list[dict]:
        """Rank KG entities by vector similarity to the query embedding."""
        vector_rank_cypher = VECTOR_RANK_CYPHER.format(
            label=f":{entity_type}" if entity_type else ""
        )

        entities = []
        remaining_budget = top_k

        for score, ids in ids_by_score.items():
            records, _, _ = self.driver.execute_query(
                vector_rank_cypher,
                ids=ids,
                query_vector=query_emb,
                limit=remaining_budget,
            )
            for record in records:
                entities.append(
                    {
                        "id": record["id"],
                        "struct_score": score,
                        "vector_sim": record["vector_sim"],
                    }
                )

            remaining_budget -= len(records)
            if remaining_budget <= 0:
                break

        return entities


driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j", "X"))
retriever = StructuralRetriever(
    model="gemma4:26b",
    driver=driver,
    ollama_embedder="qwen3-embedding:4b",
    vector_index_name="entity_index",
    fulltext_index_name="entity_fulltext_index",
    temp=0.0,
    seed=7,
)

q = "Show me some throwing equipment from the brand DSP."

answer_ids, log_dict = retriever.retrieve(q, top_k=100, entity_type="product")

print(len(answer_ids), answer_ids[:15], log_dict)
