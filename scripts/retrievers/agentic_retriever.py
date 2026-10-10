import time
from typing import Optional

import ollama
from neo4j import GraphDatabase

from .base import Retriever

PROMPT = """You are exploring a knowledge graph to find specific entities that answer complex questions.
Solve the task with interleaving Thought, Action, Observation steps. 

The following graph schema describes which edges can exist between which node types:
{schema}

When you are ready to finish the search, call the finish action.

Guidelines:
- Provide multiple options when appropriate: For queries without explicit entity mentions, aim to give users many relevant options
- Start broad, then narrow: Begin with global searches, then focus on specific neighborhoods
- Exploit seed nodes first: Try finding all entities mentioned in the query before starting neighborhood explorations
- Recognize dead ends: If local searches yield no relevant results over multiple steps, consider performing a new global search with different parameters
- Use filters strategically: Apply node_type and edge_type filters to reduce noise and focus exploration"""

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
RETURN e.id as id, labels(e) as labels, e.name as name, similarityScore"""

NEIGHBORS_CYPHER = """MATCH (start {{id: "{id}"}})-[r{edge_type}]->(neighbor)
RETURN type(r) as rel, neighbor.id as id, labels(neighbor) as labels, neighbor.name as name"""


class AgenticRetriever(Retriever):
    def __init__(
        self,
        model: str,
        standard_label: str,
        max_steps: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.standard_label = standard_label
        self.max_steps = max_steps

        self.client = ollama.Client(timeout=60)

    def global_search(
        self, search_term: str, node_type: Optional[str] = None, top_k: int = 10
    ) -> str:
        """Retrieve nodes from the knowledge graph based on semantic similarity to a search term
        
        Args:
            search_term: Keywords, entity names, or descriptive terms
            top_k: Number of nodes to return (min 10 for accuracy)
            node_type (optional): Filter by entity type, one of 'product', 'color', 'brand', 'category'
        """
        if node_type and node_type not in ["product", "color", "brand", "category"]:
            return f"Invalid node type: {node_type}. Must be one of 'product', 'color', 'brand', 'category'."
        
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
            top_k=min(10, top_k),
        )
        records, _, _ = self.driver.execute_query(vss_cypher)

        nodes_str = ""
        for record in records:
            if not node_type:
                node_labels = record["labels"]
                node_labels.remove(self.standard_label)
                node_type = node_labels[0]

            # result formatting inspired by ARK
            node_str = f"id: {record["id"]} - type: {node_type} - name: {record["name"]} - score: {record["similarityScore"]:.3f}\n"
            nodes_str += node_str

        return nodes_str

    def local_search(
        self,
        node_id: int,
        edge_type: Optional[str] = None,
    ) -> str:
        """List all 1-hop neighbors of a node
        
        Args:
            node_id: The ID of the node to explore around
            edge_type (optional): Only return neighbors connected by this type of edge, one of 'also_view', 'also_buy', 'has_category', 'has_brand', 'has_color'
        """
        if edge_type and edge_type not in ["also_view", "also_buy", "has_category", "has_brand", "has_color"]:
            return f"Invalid edge type: {edge_type}. Must be one of 'also_view', 'also_buy', 'has_category', 'has_brand', 'has_color'."

        if edge_type:
            edge_type = f":{edge_type}"
        else:
            edge_type = ""

        neighbors_cypher = NEIGHBORS_CYPHER.format(id=node_id, edge_type=edge_type)

        records, _, _ = self.driver.execute_query(neighbors_cypher)
        print(len(records))

        nodes_str = ""
        for record in records:
            node_labels = record["labels"]
            node_labels.remove(self.standard_label)
            node_type = node_labels[0]

            node_str = f"relationship: {record["rel"]} - id: {record["id"]} - name: {record["name"]} - type: {node_type}\n"
            nodes_str += node_str

        return nodes_str

    def get_document(self, node_ids: list[int]) -> str:
        """Retrieve the documents associated with a list of nodes
        
        Args:
            node_ids: A list of node IDs
        """
        str_ids = [str(node_id) for node_id in node_ids]
        Q = """
        MATCH (n)
        WHERE n.id IN $node_ids
        RETURN n.id AS id, n.document AS document
        """        
        records, _, _ = self.driver.execute_query(Q, node_ids=str_ids)
        docs_string = ""
        for record in records:
            docs_string += f"id: {record['id']}, document:\n{record['document']}\n"

        return docs_string

    def finish(self, answer_ids: list[int]) -> list[int]:
        """Return the answer and finishes the task.
        
        Args:
            answer_ids: A list of node IDs as the final answers to the query
        """
        return answer_ids

    def retrieve(self, query: str, top_k: int, entity_type: str) -> tuple[list, dict]:
        answer_ids = []
        log_dict = {}

        available_functions = {
            "global_search": self.global_search,
            "local_search": self.local_search,
            "get_document": self.get_document,
            "finish": self.finish,
        }
        tools = [self.global_search, self.local_search, self.get_document, self.finish]

        start_time = time.time()

        messages = [
            {
                "role": "system",
                "content": PROMPT.format(schema=SCHEMA),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        step = 0
        while step < self.max_steps:
            step += 1
            print(f"-----Step {step}-----")

            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    think=True,
                    options={"temperature": self.temp, "seed": self.seed},
                )
            except Exception as e:
                print(e)
                return answer_ids, log_dict
            
            # Add assistant response to history
            messages.append(response.message)

            print("Thinking:", response.message.thinking)
            print("Content:", response.message.content)

            # Process tool calls
            if not response.message.tool_calls:
                messages.append({
                    "role": "tool",
                    "content": "No tool calls made. If you want to give the final answer, call the finish action.",
                })
                continue
            for tc in response.message.tool_calls:
                if tc.function.name  in available_functions:
                    print(f"Calling {tc.function.name } with arguments: {tc.function.arguments}")
                    try:
                        result = available_functions[tc.function.name ](**tc.function.arguments)
                    except Exception as e:
                        result = f"Error executing {tc.function.name}: {e}"

                        
                    if tc.function.name == "finish":
                        return result, log_dict
                    
                    print(f"Tool Result:\n{result[:100]}\n")

                    messages.append({
                        "role": "tool",
                        "tool_name": tc.function.name,
                        "content": result,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "content": f"Unknown tool: {tc.function.name}",
                    })

 
        return [], log_dict


driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j", "Frechi2005"))
query = "What are some essential Trademark Poker brand chip sets for playing Texas Hold'em?"

retriever = AgenticRetriever(
    model="qwen3.8:27b",
    driver=driver,
    ollama_embedder="qwen3-embedding:4b",
    vector_index_name="_index",
    standard_label="entity",
    max_steps=100,
    temp=0.0,
    seed=7,
)

x = retriever.retrieve(query=query, top_k=5, entity_type="product")
print(x)