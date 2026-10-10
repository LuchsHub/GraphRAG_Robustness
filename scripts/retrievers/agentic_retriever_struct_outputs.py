import time
import yaml
from typing import Literal, Optional

from neo4j import GraphDatabase
from ollama import generate
from pydantic import BaseModel

from .base import Retriever

# Graph-CoT + ARK + GoG
PROMPT = """You are exploring a knowledge graph to find specific entities that answer complex questions.
Solve the task with interleaving Thought, Action, Observation steps. 
Thought can reason about the current situation, and Action can be three types:

The following graph schema describes which edges can exist between which node types:
{schema}

(1) global_search, which retrieves nodes based on semantic similarity to keywords.
- search_term (required): Keywords, entity names, or descriptive terms
- top_k (required): Number of nodes to return
- node_type (optional): Filter by entity type

(2) local_search, which lists 1-hop neighbors of a node.
- node_id (required): The ID of the node to explore around
- top_k (required): Number of nodes to return
- edge_type (optional): Only return neighbors connected by this type of edge
- search_term (optional): If provided, neighbors will be ranked by semantic similarity to this term

(3) finish, which returns the answer and finishes the task.
- answer_ids (required): List of node IDs as the final answer to the question

-----Question-----
{query}
{scratchpad}"""

SCHEMA = """product - also_view - product
product - also_buy - product
product - has_category - category
product - has_brand - brand
product - has_color - color"""

VSS_CYPHER = """MATCH (e:{label})
SEARCH e IN (
    VECTOR INDEX {vector_index_name}
    FOR {query_vector}
    LIMIT {top_k}
) SCORE AS similarityScore
RETURN e.id as id, labels(e) as labels, e.document as document, similarityScore"""

NEIGHBORS_CYPHER = """MATCH (start {{id: "{id}"}})-[r{edge_type}]->(neighbor)
RETURN type(r) as rel, neighbor.id as id, labels(neighbor) as labels, neighbor.document as document"""

QUERY_FILTERED_NEIGHBORS_CYPHER = """MATCH (start {{id: "{id}"}})-[r{edge_type}]->(neighbor)
WHERE neighbor.embedding IS NOT NULL
WITH r, neighbor, vector.similarity.cosine(neighbor.embedding, {query_vector}) AS similarityScore
ORDER BY similarityScore DESC
LIMIT {top_k}
RETURN type(r) as rel, neighbor.id as id, labels(neighbor) as labels, neighbor.document as document, similarityScore"""

class global_search(BaseModel):
    action_type: Literal["global_search"] = "global_search"
    search_term: str
    top_k: int
    node_type: Optional[Literal["product", "category", "brand", "color"]]

class local_search(BaseModel):
    action_type: Literal["local_search"] = "local_search"
    node_id: int
    top_k: int
    edge_type: Optional[Literal["also_view", "also_buy", "has_category", "has_brand", "has_color"]]
    search_term: Optional[str]

class finish(BaseModel):
    action_type: Literal["finish"] = "finish"
    answer_ids: list[int]

class Step(BaseModel):
    thought: str
    action: global_search | local_search | finish


class AgenticRetriever(Retriever):
    def __init__(
        self,
        model: str,
        fulltext_index_name: str,
        standard_label: str,
        max_steps: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.fulltext_index_name = fulltext_index_name
        self.standard_label = standard_label
        self.max_steps = max_steps

    def global_search(
        self, search_term: str, node_type: Optional[str], top_k: int = 5
    ) -> str:
        query_emb = self.ollama_embedder.embed_query(
            search_term,
            options={"temperature": self.temp, "seed": self.seed},
        )

        if node_type:
            label = node_type
            vector_index_name = f"{node_type}{self.vector_index_name}"
        else:
            label = self.standard_label
            vector_index_name = f"{self.standard_label}{self.vector_index_name}"

        vss_cypher = VSS_CYPHER.format(
            label=label,
            vector_index_name=vector_index_name,
            query_vector=query_emb,
            top_k=top_k,
        )
        records, _, _ = self.driver.execute_query(vss_cypher)

        nodes_str = ""
        for record in records:
            if not node_type:
                node_labels = record["labels"]
                node_labels.remove(self.standard_label)
                node_type = node_labels[0]

            # result formatting inspired by ARK
            node_str = f"id: {record["id"]} - type: {node_type} - score: {record["similarityScore"]:.3f} - document:\n{record["document"]}\n"
            nodes_str += node_str

        return nodes_str

    def local_search(
        self,
        node_id: int,
        edge_type: Optional[str],
        search_term: Optional[str],
        top_k: int = 5,
    ) -> str:
        if edge_type:
            edge_type = f":{edge_type}"
        else:
            edge_type = ""
        if search_term:
            query_emb = self.ollama_embedder.embed_query(
                search_term,
                options={"temperature": self.temp, "seed": self.seed},
            )
            neighbors_cypher = QUERY_FILTERED_NEIGHBORS_CYPHER.format(
                id=node_id, edge_type=edge_type, query_vector=query_emb, top_k=top_k
            )
        else:
            neighbors_cypher = NEIGHBORS_CYPHER.format(id=node_id, edge_type=edge_type)

        records, _, _ = self.driver.execute_query(neighbors_cypher)
        if not search_term and len(records) > top_k:
            return f"Found {len(records)} neighbors, which is higher than your specified top_k {top_k}. Please provide a search_term to select the top-k most relevant neighbors or allow a higher top_k."

        nodes_str = ""
        for record in records:
            node_labels = record["labels"]
            node_labels.remove(self.standard_label)
            node_type = node_labels[0]

            if search_term:
                node_str = f"relationship: {record["rel"]} - id: {record["id"]} - type: {node_type} - score: {record["similarityScore"]:.3f} - document:\n{record["document"]}\n"
            else:
                node_str = f"relationship: {record["rel"]} - id: {record["id"]} - type: {node_type} - document:\n{record["document"]}\n"
            nodes_str += node_str

        return nodes_str

    def retrieve(self, query: str, top_k: int, entity_type: str) -> tuple[list, dict]:
        answer_ids = []
        log_dict = {}

        start_time = time.time()

        i = 0
        scratchpad = ""
        while i < self.max_steps:
            prompt = PROMPT.format(schema=SCHEMA, query=query, scratchpad=scratchpad)
            response = generate(
                prompt=prompt,
                model="gemma4:26b",
                think=False,
                format=Step.model_json_schema(),
            )
            step = Step.model_validate_json(response.response)
            print(step)

            # append Thought
            scratchpad += f"-----Thought {i+1}-----\n{step.thought}\n"

            tool_name = step.action.__class__.__name__
            if tool_name == "finish":
                return step.action.answer_ids, PROMPT.format(
                    schema=SCHEMA, query=query, scratchpad=scratchpad
                )
            args = step.action.model_dump()
            args.pop("action_type")
            scratchpad += f"-----Action {i+1}-----\n{tool_name} with args: {args}\n"

            result = getattr(self, tool_name)(**args)

            # append Observation
            scratchpad += f"-----Observation {i+1}-----\n{result}\n"
            print(f"-----Observation {i+1}-----\n{result[:100]}\n")

            i += 1

        return [], PROMPT.format(query=query, scratchpad=scratchpad)


driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j", "X"))
query = "Show me tortoise colored products that are frequently bought with items from the frame-mounted pumps category."

retriever = AgenticRetriever(
    model="gemma4:26b",
    driver=driver,
    ollama_embedder="qwen3-embedding:4b",
    vector_index_name="_index",
    fulltext_index_name="_fulltext_index",
    standard_label="entity",
    max_steps=10,
    temp=0.0,
    seed=7,
)

answer_ids, log_dict = retriever.retrieve(query=query, top_k=5, entity_type="product")
print(answer_ids)
