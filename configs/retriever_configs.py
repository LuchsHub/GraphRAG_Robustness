from enum import Enum
from typing import Any

from pydantic import BaseModel

ENTITY_TYPES_CYPHER = """CALL db.labels() YIELD label 
WHERE label <> $general_label
RETURN collect(label) AS entity_types"""

REL_TYPES_CYPHER = """CALL db.relationshipTypes() YIELD relationshipType
RETURN collect(relationshipType) AS relation_types"""

POSSIBLE_RELATIONS_CYPHER = """MATCH (a)-[r]-(b)
WITH 
    [l IN labels(a) WHERE l <> $general_label][0] AS src_type,
    type(r) AS rel_type,
    [l IN labels(b) WHERE l <> $general_label][0] AS dst_type
RETURN DISTINCT src_type, rel_type, dst_type"""

VECTOR_SEARCH_CYPHER = """MATCH (e:{label})
SEARCH e IN (
    VECTOR INDEX {vector_index_name}
    FOR {query_vector}
    LIMIT {top_k}
) SCORE AS similarityScore
RETURN e.id as id, similarityScore"""


class VSSRetrieverConfig:
    vector_search_cypher = VECTOR_SEARCH_CYPHER


class StructuralRetrieverConfig:
    entity_types_cypher = ENTITY_TYPES_CYPHER

    ner_prompt = """Given a list of entity types, identify all entities of those types from the query.

    For each identified entity, extract the following information:
    - name: Name of the entity.
    - type: One of the following types: {entity_types}

    Query: {query}

    Output: """
    fulltext_search_cypher = """CALL db.index.fulltext.queryNodes("{fulltext_index_name}", "'{mention_name}'") 
    YIELD node
    WHERE toLower(node.name) = toLower("{mention_name}")
    RETURN node.id as id"""
    vector_search_cypher = VECTOR_SEARCH_CYPHER
    hops = 2
    h_hop_neighbors_cypher = """MATCH (start {{id: "{id}"}})-[*{h}]->(neighbor:{answer_entity_type})
    RETURN collect(DISTINCT neighbor.id) AS ids"""
    vector_rank_cypher = """MATCH (p)
    WHERE p.id IN $ids
    RETURN p.id AS id, vector.similarity.cosine(p.embedding, $query_vector) AS vector_sim
    ORDER BY vector_sim DESC
    LIMIT $limit"""

    def get_ner_output_schema(entity_types: list[str]) -> dict[str, Any]:
        DynamicTypeEnum = Enum("DynamicTypeEnum", {t: t for t in entity_types})

        class Entity(BaseModel):
            name: str
            type: DynamicTypeEnum

        class Entities(BaseModel):
            entities: list[Entity]

        return Entities.model_json_schema()


class AgenticRetrieverConfig:
    entity_types_cypher = ENTITY_TYPES_CYPHER
    relation_types_cypher = REL_TYPES_CYPHER
    possible_relations_cypher = POSSIBLE_RELATIONS_CYPHER

    system_prompt = """You are exploring a knowledge graph to find specific entities that answer complex questions.

    The following graph schema describes which edges can exist between which node types:
    {schema}

    When you are ready to finish the search, call the finish action. 
    Do not provide conversational responses, only use tool calls."""

    def get_tool_schemas(
        entity_types: list[str], relation_types: list[str], top_k: int
    ) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "global_search",
                    "description": "Retrieve nodes from the knowledge graph based on semantic similarity to a search term",
                    "parameters": {
                        "type": "object",
                        "required": ["search_term", "top_k"],
                        "properties": {
                            "search_term": {
                                "type": "string",
                                "description": "Keywords, entity names, or descriptive terms",
                            },
                            "node_type": {
                                "type": "string",
                                "enum": entity_types,
                                "description": f"Filter by entity type, one of {entity_types}",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of nodes to return (min 10 for accuracy)",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "local_search",
                    "description": "List all 1-hop neighbors of a node",
                    "parameters": {
                        "type": "object",
                        "required": ["node_id"],
                        "properties": {
                            "node_id": {
                                "type": "integer",
                                "description": "ID of the node to explore around",
                            },
                            "edge_type": {
                                "type": "string",
                                "enum": relation_types,
                                "description": f"Only return neighbors connected by this type of edge, one of {relation_types}",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_documents",
                    "description": "Retrieve the documents associated with a list of nodes",
                    "parameters": {
                        "type": "object",
                        "required": ["node_ids"],
                        "properties": {
                            "node_ids": {
                                "type": "array",
                                "description": "List of node IDs",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "Return the answer and finishes the task",
                    "parameters": {
                        "type": "object",
                        "required": ["answer_ids"],
                        "properties": {
                            "answer_ids": {
                                "type": "array",
                                "description": f"List of node IDs (max {top_k}) as the final answers to the query",
                            },
                        },
                    },
                },
            },
        ]

    vss_cypher = """MATCH (e:{label})
    SEARCH e IN (
        VECTOR INDEX {vector_index_name}
        FOR {query_vector}
        LIMIT {top_k}
    ) SCORE AS similarityScore
    RETURN e.id as id, labels(e) as labels, e.name as name, similarityScore"""

    neighbors_cypher = """MATCH (start {{id: "{id}"}})-[r{edge_type}]->(neighbor)
    RETURN type(r) as rel, neighbor.id as id, labels(neighbor) as labels, neighbor.name as name"""

    get_docs_cypher = """MATCH (n)
    WHERE n.id IN $node_ids
    RETURN n.id AS id, n.document AS document"""

    max_tool_output_length = 100000
    max_steps = 10
