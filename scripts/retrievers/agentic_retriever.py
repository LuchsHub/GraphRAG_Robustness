import time
from typing import Optional
import ollama
from neo4j import GraphDatabase

from configs.retriever_configs import AgenticRetrieverConfig
from .base import Retriever


class AgenticRetriever(Retriever):
    def __init__(
        self,
        model: str,
        graph_schema: str,
        standard_label: str,
        entity_types: list[str],
        relation_types: list[str],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.graph_schema = graph_schema
        self.standard_label = standard_label
        self.entity_types = entity_types
        self.relation_types = relation_types

        self.client = ollama.Client(timeout=60)

    def global_search(
        self, search_term: str, top_k: int, node_type: Optional[str] = None
    ) -> str:
        if node_type and node_type not in self.entity_types:
            return (
                f"Invalid node type: {node_type}. Must be one of: {self.entity_types}"
            )

        query_emb = self.ollama_embedder.embed_query(
            search_term,
            options={"temperature": self.temp, "seed": self.seed},
        )

        # determine target label and vector index
        # search for "product" in "product_index" (specific)
        if node_type:
            label = node_type
            vector_index_name = f"{node_type}{self.vector_index_name_appendix}"
        # search for "entity" in "entity_index" (all nodes)
        else:
            label = self.standard_label
            vector_index_name = f"{self.standard_label}{self.vector_index_name_appendix}"

        vss_cypher = AgenticRetrieverConfig.vss_cypher.format(
            label=label,
            vector_index_name=vector_index_name,
            query_vector=query_emb,
            top_k=min(10, top_k),
        )
        records, _, _ = self.driver.execute_query(vss_cypher)

        nodes_str = ""
        for record in records:
            # if no node type was specified, get it from the node's labels
            if not node_type:
                node_labels = record["labels"]
                node_labels.remove(self.standard_label)
                node_type = node_labels[0]

            node_str = f"id: {record["id"]} - type: {node_type} - name: {record["name"]} - score: {record["similarityScore"]:.3f}\n"
            nodes_str += node_str

        return nodes_str

    def local_search(
        self,
        node_id: int,
        edge_type: Optional[str] = None,
    ) -> str:
        if edge_type and edge_type not in self.relation_types:
            return (
                f"Invalid edge type: {edge_type}. Must be one of: {self.relation_types}"
            )

        edge_filter = f":{edge_type}" if edge_type else ""
        neighbors_cypher = AgenticRetrieverConfig.neighbors_cypher.format(
            id=node_id, edge_type=edge_filter
        )

        records, _, _ = self.driver.execute_query(neighbors_cypher)

        nodes_str = ""
        for record in records:
            node_labels = record["labels"]
            node_labels.remove(self.standard_label)
            resolved_type = node_labels[0] if node_labels else "unknown"

            node_str = f"relationship: {record["rel"]} - id: {record["id"]} - type: {resolved_type} - name: {record["name"]}\n"
            nodes_str += node_str

        return nodes_str

    def get_documents(self, node_ids: list[int]) -> str:
        str_ids = [str(node_id) for node_id in node_ids]
        records, _, _ = self.driver.execute_query(
            AgenticRetrieverConfig.get_docs_cypher, node_ids=str_ids
        )

        docs_string = ""
        for record in records:
            docs_string += f"id: {record["id"]}, document:\n{record["document"]}\n"

        return docs_string

    def finish(self, answer_ids: list[int]) -> list[int]:
        return answer_ids

    def retrieve(self, query: str, top_k: int, entity_type: str) -> tuple[list, dict]:
        tool_schemas = AgenticRetrieverConfig.get_tool_schemas(
            entity_types=self.entity_types, relation_types=self.relation_types
        )

        available_functions = {}
        for tool in tool_schemas:
            available_functions[tool["function"]["name"]] = getattr(
                self, tool["function"]["name"]
            )

        messages = [
            {
                "role": "system",
                "content": AgenticRetrieverConfig.system_prompt.format(
                    schema=self.graph_schema
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        step = 0
        log_dict = {}
        while step < AgenticRetrieverConfig.max_steps:
            step += 1
            print(f"-----Step {step}-----")

            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    think="high",
                    tools=tool_schemas,
                    options={"temperature": self.temp, "seed": self.seed},
                )
            except Exception as e:
                return [], log_dict

            messages.append(response.message)
            print("Thinking: ", response.message.thinking)
            print("Content: ", response.message.content)

            if not response.message.tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "content": "No tool calls made. If you want to give the final answer, call the finish action.",
                    }
                )
                continue

            for tc in response.message.tool_calls:
                if tc.function.name in available_functions:
                    print(
                        f"Calling {tc.function.name} with arguments: {tc.function.arguments}"
                    )
                    try:
                        result = available_functions[tc.function.name](
                            **tc.function.arguments
                        )
                    except Exception as e:
                        result = f"Error: {e}"

                    if tc.function.name == "finish":
                        return result[:top_k], log_dict

                    print(result[:100])

                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": tc.function.name,
                            "content": str(result),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "tool",
                            "content": f"Unknown tool: {tc.function.name}",
                        }
                    )

        return [], log_dict


driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j", "X"))
GRAPH_SCHEMA = """product - also_view - product
product - also_buy - product
product - has_category - category
product - has_brand - brand
product - has_color - color"""
QUERY = "Show me some throwing equipment from the brand DSP."

retriever = AgenticRetriever(
    model="qwen3.8:27b",
    driver=driver,
    ollama_embedder="qwen3-embedding:4b",
    vector_index_name_appendix="_index",
    standard_label="entity",
    entity_types=["product", "brand", "category", "color"],
    relation_types=["also_view", "also_buy", "has_category", "has_brand", "has_color"],
    graph_schema=GRAPH_SCHEMA,
    temp=0.0,
    seed=7,
)

x = retriever.retrieve(query=QUERY, top_k=5, entity_type="product")
print(x)
