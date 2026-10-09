import time
from typing import Literal, Optional

from neo4j import GraphDatabase
from ollama import generate
from pydantic import BaseModel

from .base import Retriever

# Graph-CoT + ARK + GoG
PROMPT = """You are exploring a knowledge graph to find specific entities that answer complex questions.
Solve the task with interleaving Thought, Action, Observation steps. 
Thought can reason about the current situation, and Action can be three types:

(1) global_search, which retrieves nodes based on semantic similarity to a query.
- query (required): Keywords, entity names, or descriptive terms
- top_k (optional): Number of results to return

(2) local_search, which lists the neighbors of a node in the graph.
- node_id (required): The ID of the node to explore around
- query (optional): If provided, neighbors will be ranked by semantic similarity of their description to this query
- top_k (optional): Number of results to display

(3) finish, which returns the answer and finishes the task.
- answer_ids (required): List of node IDs as the final answer to the question

-----Question-----
{query}
{scratchpad}"""

SEARCH_CYPHER = """MATCH (e)
SEARCH e IN (
    VECTOR INDEX {vector_index_name}
    FOR {query_vector}
    LIMIT {top_k}
) SCORE AS similarityScore
RETURN e.id as id, labels(e) as labels, e.document as document, similarityScore"""

NEIGHBORS_CYPHER = """MATCH (start {{id: "{id}"}})-[r]->(neighbor)
RETURN type(r) as rel, neighbor.id as id, labels(neighbor) as labels, neighbor.document as document"""

QUERY_FILTERED_NEIGHBORS_CYPHER = """MATCH (start {{id: "{id}"}})-[r]->(neighbor)
WHERE neighbor.embedding IS NOT NULL
WITH r, neighbor, vector.similarity.cosine(neighbor.embedding, {query_vector}) AS similarityScore
ORDER BY similarityScore DESC
LIMIT {top_k}
RETURN type(r) as rel, neighbor.id as id, labels(neighbor) as labels, neighbor.document as document, similarityScore"""


class Step(BaseModel):
    thought: str
    action: Literal["global_search", "local_search", "finish"]
    query: Optional[str]
    node_id: Optional[int]
    top_k: int
    answer_ids: list[int]


class AgenticRetriever(Retriever):
    def __init__(
        self, model: str, fulltext_index_name: str, max_steps: int, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.fulltext_index_name = fulltext_index_name
        self.max_steps = max_steps

    def global_search(self, query: str, top_k: int) -> str:
        query_emb = self.ollama_embedder.embed_query(
            query,
            options={"temperature": self.temp, "seed": self.seed},
        )

        search_cypher = SEARCH_CYPHER.format(
            vector_index_name=self.vector_index_name,
            query_vector=query_emb,
            top_k=top_k,
        )
        records, _, _ = self.driver.execute_query(search_cypher)

        nodes_str = ""
        for record in records:
            node_labels = record["labels"]
            node_labels.remove("entity")
            node_type = node_labels[0]

            # result formatting inspired by ARK
            node_str = f"id: {record["id"]} - type: {node_type} - score: {record["similarityScore"]:.3f} - document:\n{record["document"]}\n"
            nodes_str += node_str

        return nodes_str

    def local_search(self, node_id: int, query: Optional[str], top_k: int) -> str:
        if query:
            query_emb = self.ollama_embedder.embed_query(
                query,
                options={"temperature": self.temp, "seed": self.seed},
            )
            neighbors_cypher = QUERY_FILTERED_NEIGHBORS_CYPHER.format(
                id=node_id, query_vector=query_emb, top_k=top_k
            )
        else:
            neighbors_cypher = NEIGHBORS_CYPHER.format(id=node_id)

        nodes_str = ""
        records, _, _ = self.driver.execute_query(neighbors_cypher)
        for record in records:
            node_labels = record["labels"]
            node_labels.remove("entity")
            node_type = node_labels[0]

            if query:
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
            prompt = PROMPT.format(query=query, scratchpad=scratchpad)
            response = generate(
                prompt=prompt,
                model="gemma4:26b",
                think=False,
                format=Step.model_json_schema(),
                options={"temperature": self.temp, "seed": self.seed},
            )
            step = Step.model_validate_json(response.response)

            # append Thought
            scratchpad += f"-----Thought {i+1}-----\n{step.thought}\n"
            print(f"-----Thought {i+1}-----\n{step.thought}\n")

            # append and execute Action
            if step.action == "finish":
                scratchpad += (
                    f"-----Action {i+1}-----\nfinish(answer_ids={step.answer_ids})\n"
                )
                print(f"-----Action {i+1}-----\nfinish(answer_ids={step.answer_ids})\n")
                return step.answer_ids, PROMPT.format(
                    query=query, scratchpad=scratchpad
                )
            elif step.action == "global_search":
                scratchpad += f"-----Action {i+1}-----\nglobal_search(query={step.query}, top_k={step.top_k})\n"
                print(f"-----Action {i+1}-----\nglobal_search(query={step.query}, top_k={step.top_k})\n")
                result = self.global_search(query=step.query, top_k=step.top_k)
            elif step.action == "local_search":
                scratchpad += f"-----Action {i+1}-----\nlocal_search(node_id={step.node_id}, query={step.query}, top_k={step.top_k})\n"
                print(f"-----Action {i+1}-----\nlocal_search(node_id={step.node_id}, query={step.query}, top_k={step.top_k})\n")
                result = self.local_search(
                    node_id=step.node_id, query=step.query, top_k=step.top_k
                )

            # append Observation
            scratchpad += f"-----Observation {i+1}-----\n{result}\n"
            print(f"-----Observation {i+1}-----\n{result}\n")

            i += 1

        return [], PROMPT.format(query=query, scratchpad=scratchpad)


driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j", "X"))
query = "Can I find a SeaStar Pro Rack Steering Kit with stainless steel cable output ends?"

retriever = AgenticRetriever(
    model="gemma4:26b",
    driver=driver,
    ollama_embedder="qwen3-embedding:4b",
    vector_index_name="entity_index",
    fulltext_index_name="entity_fulltext_index",
    max_steps=10,
    temp=0.0,
    seed=7,
)

answer_ids, log_dict = retriever.retrieve(query=query, top_k=5, entity_type="product")
print(log_dict)
print(answer_ids)
