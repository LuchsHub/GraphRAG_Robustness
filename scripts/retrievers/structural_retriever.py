from ollama import generate
from pydantic import BaseModel

from configs.retriever_configs import StructuralRetrieverConfig
from .base import Retriever


class Entity(BaseModel):
    name: str
    type: str


class Entities(BaseModel):
    entities: list[Entity]


class StructuralRetriever(Retriever):
    def __init__(
        self,
        standard_label: str,
        model: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model

        self.entity_types = self.driver.execute_query(
            StructuralRetrieverConfig.entity_types_cypher, general_label=standard_label
        )[0][0]["entity_types"]

    def retrieve(
        self, query: str, top_k: int, answer_entity_type: str
    ) -> tuple[list, dict]:
        answer_ids = []
        log_dict = {}

        # vectorize query
        query_emb = self.ollama_embedder.embed_query(
            query,
            options={"temperature": self.temp, "seed": self.seed},
        )

        # NER with a shortened version of the PathRAG NER prompt
        mentioned_entities = self.extract_entities_from_query(query)

        # link every entity to KG
        seeds = []
        log_dict["seed_entities"] = []
        for entity in mentioned_entities.entities:
            id, used_vss = self.link_entity_to_graph(entity)
            if (id, entity.type) not in seeds:
                seeds.append((id, entity.type))
            log_dict["seed_entities"].append(
                {
                    "mention_name": entity.name,
                    "mention_type": entity.type,
                    "linked_id": id,
                    "used_vss": used_vss,
                }
            )

        # VSS fallback
        if not seeds:
            query_entity_id = self.find_query_entity(query_emb, answer_entity_type)
            log_dict["query_entity_id"] = query_entity_id
            seeds.append((query_entity_id, answer_entity_type))

        # score and group by structural score, inspired by HippoRAG
        ids_by_score = self.compute_structural_scores(seeds, answer_entity_type)

        # break ties with vector similarity, return top_k
        ranked_entities = self.rank_by_vector_similarity(ids_by_score, query_emb, top_k)

        answer_ids = [entity["id"] for entity in ranked_entities]
        log_dict["scores"] = {
            entity["id"]: (entity["struct_score"], entity["vector_sim"])
            for entity in ranked_entities
        }

        return answer_ids, log_dict

    def extract_entities_from_query(self, query: str) -> Entities:
        """Extract mentioned entities from the user query."""
        ner_prompt = StructuralRetrieverConfig.ner_prompt.format(
            query=query, entity_types=self.entity_types
        )
        ner_output_schema = StructuralRetrieverConfig.get_ner_output_schema(
            entity_types=self.entity_types
        )
        response = generate(
            model=self.model,
            prompt=ner_prompt,
            options={"seed": self.seed, "temperature": self.temp},
            think=False,
            format=ner_output_schema,
        )

        return Entities.model_validate_json(response.response)

    def link_entity_to_graph(self, entity: Entity) -> tuple[str, bool]:
        """Link an entity to the knowledge graph through exact-match name search, alternatively through document-based VSS."""

        # try case-insensitive exact-matching
        exact_match_cypher = StructuralRetrieverConfig.exact_match_cypher.format(
            label=entity.type
        )
        records, _, _ = self.driver.execute_query(exact_match_cypher, name=entity.name)
        if records:
            return records[0]["id"], False

        # resort to VSS
        else:
            entity_emb = self.ollama_embedder.embed_query(
                entity.name,
                options={"temperature": self.temp, "seed": self.seed},
            )
            vector_index_name = f"{entity.type}{self.vector_index_name_appendix}"
            vector_search_cypher = (
                StructuralRetrieverConfig.vector_search_cypher.format(
                    label=entity.type,
                    vector_index_name=vector_index_name,
                )
            )
            records, _, _ = self.driver.execute_query(
                vector_search_cypher, query_vector=entity_emb, top_k=5
            )
            return records[0]["id"], True

    def find_query_entity(self, query_emb: list[float], answer_entity_type: str) -> str:
        """Find a KG entity based on the query embedding."""
        vector_index_name = f"{answer_entity_type}{self.vector_index_name_appendix}"
        vector_search_cypher = StructuralRetrieverConfig.vector_search_cypher.format(
            label=answer_entity_type,
            vector_index_name=vector_index_name,
        )
        records, _, _ = self.driver.execute_query(
            vector_search_cypher, query_vector=query_emb, top_k=5
        )
        return records[0]["id"]

    def compute_structural_scores(
        self, seeds: list[tuple[str, str]], answer_entity_type: str
    ) -> dict[float, list[str]]:
        """Add scores to KG entities based on their distance to seed entities."""
        # set score = 1 for target type seeds, since they themselves might be answers
        scores = {
            seed_id: 1
            for seed_id, seed_type in seeds
            if seed_type == answer_entity_type
        }

        for seed_id, _ in seeds:
            visited = set()
            visited.add(seed_id)

            for hop in range(1, StructuralRetrieverConfig.hops + 1):
                score_add = 1 / (hop + 1)
                # get all exactly-h-hop neighbors
                neighbors_cypher_query = (
                    StructuralRetrieverConfig.h_hop_neighbors_cypher.format(
                        h=hop, answer_entity_type=answer_entity_type
                    )
                )
                records, _, _ = self.driver.execute_query(
                    neighbors_cypher_query, id=seed_id
                )

                # add score based on HOP_SCORING
                for neighbor in records[0]["ids"]:
                    if neighbor not in visited:
                        scores[neighbor] = scores.get(neighbor, 0) + score_add
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
        self, ids_by_score: dict[float, list[str]], query_emb: list[float], top_k: int
    ) -> list[dict]:
        """Rank KG entities by vector similarity to the query embedding."""
        entities = []
        remaining_budget = top_k

        for score, ids in ids_by_score.items():
            records, _, _ = self.driver.execute_query(
                StructuralRetrieverConfig.vector_rank_cypher,
                ids=ids,
                query_vector=query_emb,
                limit=remaining_budget,
            )
            for record in records:
                entities.append(
                    {
                        "id": int(record["id"]),
                        "struct_score": round(score, 3),
                        "vector_sim": round(record["vector_sim"], 3),
                    }
                )

            remaining_budget -= len(records)
            if remaining_budget <= 0:
                break

        return entities
