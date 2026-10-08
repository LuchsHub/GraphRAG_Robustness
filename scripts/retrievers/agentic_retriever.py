import time
from typing import Optional

from neo4j import GraphDatabase
from ollama import chat

from .base import Retriever


# Graph-CoT + ARK + GoG
PROMPT = """You are exploring a knowledge graph to find specific entities that answer complex questions.
Solve the task with interleaving Thought, Action, Observation steps. 
Thought can reason about the current situation, and Action can be three types:

(1) RetrieveNode, which retrieves related nodes from the graph according to the corresponding query.
- query (required): Keywords, entity names, or descriptive terms
- size (required): Number of results to return

(2) NeighbourCheck, which lists the neighbours of the node in the graph and returns them.
- node_id (required): The ID of the node to explore around
- query (optional): Keywords to filter neighborhood results

(3) Finish, which returns the answer and finishes the task.
- answer_ids (required): List of node IDs as the final answer to the question"""

def RetrieveNode(query: str, size: int):
    pass

def NeighbourCheck(node_id: int, query: Optional[str] = None):
    pass

def Finish(answer_ids: list[int]):
    pass


class AgenticRetriever(Retriever):
    def __init__(self, model: str, fulltext_index_name: str, max_steps: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.fulltext_index_name = fulltext_index_name
        self.max_steps = max_steps

    def retrieve(self, query: str, top_k: int, entity_type: str) -> tuple[list, dict]:
        answer_ids = []
        log_dict = {}

        start_time = time.time()

        messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": query}]

        i = 0
        while i < self.max_steps:
            # use native thinking for Thought step + tool calling API for Action step
            response = chat(
                model="gemma4:26b",
                messages=messages,
                think="high",
                tools=[RetrieveNode, NeighbourCheck, Finish],
                options={"temperature": self.temp, "seed": self.seed},
            )

            print(f"Thinking {i+1}: {response.message.thinking}")
            print(f"Action {i+1}: {response.message.tool_calls[0].function.name}")

            """ messages.append({"role": "assistant", "content": f"Thought {i+1}: {step.thought}"})

            tool_name = type(step.action).__name__
            params_str = ", ".join(f"{k}={v}" for k, v in step.action.model_dump().items())

            print(tool_name, params_str) """





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
