import time
from typing import Optional

from neo4j import GraphDatabase
from ollama import chat
from pydantic import BaseModel

from .base import Retriever

# Graph-CoT + ARK + GoG
PROMPT = """You are exploring a knowledge graph to find specific entities that answer complex questions.
Solve the task with interleaving Thought, Action, Observation steps. 
Thought can reason about the current situation, and Action can be three types:

(1) retrieve_nodes, which retrieves related nodes from the graph according to the corresponding query.
- query (required): Keywords, entity names, or descriptive terms
- size (required): Number of results to return

(2) check_neighbors, which lists the neighbours of the node in the graph and returns them.
- node_id (required): The ID of the node to explore around
- query (optional): Keywords to filter neighborhood results

(3) finish, which returns the answer and finishes the task.
- answer_ids (required): List of node IDs as the final answer to the question"""


class retrieve_nodes(BaseModel):
    query: str
    size: int


class check_neighbors(BaseModel):
    node_id: int
    query: Optional[str]


class finish(BaseModel):
    answer_ids: list[int]


class Step(BaseModel):
    thought: str
    action: retrieve_nodes | check_neighbors | finish


class AgenticRetriever(Retriever):
    def __init__(
        self, model: str, fulltext_index_name: str, max_steps: int, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.fulltext_index_name = fulltext_index_name
        self.max_steps = max_steps

    def retrieve(self, query: str, top_k: int, entity_type: str) -> tuple[list, dict]:
        answer_ids = []
        log_dict = {}

        start_time = time.time()

        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": query},
        ]

        i = 0
        while i < self.max_steps:
            # use native thinking for Thought step + tool calling API for Action step
            response = chat(
                model="gemma4:26b",
                messages=messages,
                think=False,
                format=Step.model_json_schema(),
                options={"temperature": self.temp, "seed": self.seed},
            )
            step = Step.model_validate_json(response.message.content)
            messages.append(
                {"role": "assistant", "content": f"Thought {i+1}: {step.thought}"}
            )
            tool_name = type(step.action).__name__
            params_str = "".join(
                f"{k}={v}" for k, v in step.action.model_dump().items()
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Action {i+1}: {tool_name}({params_str})",
                }
            )

            if isinstance(step.action, retrieve_nodes):
                # Handle retrieve_nodes action
                pass
            elif isinstance(step.action, check_neighbors):
                # Handle check_neighbors action
                pass
            elif isinstance(step.action, finish):
                # Handle finish action
                pass

            i += 1

        return messages


driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j", "X"))
query = "I'm looking for a high-quality, USA-made pocket knife with a high carbon steel blade, nickel silver bolsters, and a smooth delrin handle that is approximately 3 7/8 inches when closed."

retriever = AgenticRetriever(
    model="gemma4:26b",
    driver=driver,
    ollama_embedder="qwen3-embedding:4b",
    vector_index_name="entity_index",
    fulltext_index_name="entity_fulltext_index",
    max_steps=1,
    temp=0.0,
    seed=7,
)
answer_ids = retriever.retrieve(query=query, top_k=5, entity_type="product")

for msg in answer_ids:
    print(msg)
