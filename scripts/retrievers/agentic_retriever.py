from ollama import chat
from pydantic import BaseModel
from typing import Optional


class RetrieveNode(BaseModel):
    query: str
    size: int


class NeighbourCheck(BaseModel):
    node_id: int
    query: Optional[str]


class Finish(BaseModel):
    answer_ids: list[int]


class Step(BaseModel):
    thought: str
    action: RetrieveNode | NeighbourCheck | Finish


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
- answer_ids (required): List of node IDs as the final answer to the question

Question: {question}"""

query = PROMPT.format(
    question="Is there an Aminco brand pin for an NFL helmet that you could recommend?"
)
messages = [{"role": "user", "content": query}]

response = chat(
    model="gemma4:26b",
    messages=messages,
    think="high",
    format=Step.model_json_schema(),
)
step = Step.model_validate_json(response.message.content)
print(step)
print(type(step.action))
