from typing import Optional

import ollama

from configs.retriever_configs import AgenticRetrieverConfig
from .base import Retriever


class AgenticRetriever(Retriever):
    def __init__(
        self,
        model: str,
        standard_label: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.standard_label = standard_label

        self.client = ollama.Client()
        self.entity_types = self.driver.execute_query(
            AgenticRetrieverConfig.entity_types_cypher, general_label=standard_label
        )[0][0]["entity_types"]
        self.rel_types = self.driver.execute_query(
            AgenticRetrieverConfig.relation_types_cypher
        )[0][0]["relation_types"]

        records, _, _ = self.driver.execute_query(
            AgenticRetrieverConfig.possible_relations_cypher,
            general_label=standard_label,
        )
        schema_str = ""
        for record in records:
            schema_str += (
                f"{record["src_type"]} - {record["rel_type"]} - {record["dst_type"]}\n"
            )
        self.graph_schema = schema_str.rstrip("\n")

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
            vector_index_name = (
                f"{self.standard_label}{self.vector_index_name_appendix}"
            )

        vss_cypher = AgenticRetrieverConfig.vss_cypher.format(
            label=label,
            vector_index_name=vector_index_name,
        )
        records, _, _ = self.driver.execute_query(
            vss_cypher,
            query_vector=query_emb,
            top_k=min(10, top_k),
        )

        nodes_str = ""
        for record in records:
            node_str = f"id: {record["id"]} - name: {record["name"]} - score: {record["similarityScore"]:.3f}\n"
            nodes_str += node_str

        return nodes_str

    def local_search(
        self,
        node_id: int,
        edge_type: Optional[str] = None,
    ) -> str:
        if edge_type and edge_type not in self.rel_types:
            return f"Invalid edge type: {edge_type}. Must be one of: {self.rel_types}"

        edge_filter = f":{edge_type}" if edge_type else ""
        neighbors_cypher = AgenticRetrieverConfig.neighbors_cypher.format(
            edge_type=edge_filter
        )

        records, _, _ = self.driver.execute_query(neighbors_cypher, id=node_id)

        nodes_str = ""
        for record in records:
            node_str = f"relationship: {record["rel"]} - id: {record["id"]} - name: {record["name"]}\n"
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
        return [int(id) for id in answer_ids]

    def retrieve(
        self, query: str, top_k: int, answer_entity_type: str
    ) -> tuple[list, dict]:
        answer_ids = []
        log_dict = {"steps": []}
        finished = False

        tool_schemas = AgenticRetrieverConfig.get_tool_schemas(
            entity_types=self.entity_types, relation_types=self.rel_types, top_k=top_k
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

        all_actions = []
        step = 0
        total_eval_count = 0
        while step < AgenticRetrieverConfig.max_steps:
            step += 1

            if step == AgenticRetrieverConfig.max_steps:
                messages.append(
                    {
                        "role": "system",
                        "content": "Last step. You must call the finish action now.",
                    }
                )

            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    think="high",
                    tools=tool_schemas,
                    options={"temperature": self.temp, "seed": self.seed},
                )
            except Exception as e:
                log_dict["generation_error"] = str(e)
                break

            messages.append(response.message)
            total_eval_count += response.eval_count

            if not response.message.tool_calls:
                messages.append(
                    {
                        "role": "system",
                        "content": "No tool calls made. If you want to give the final answer, call the finish action.",
                    }
                )
                continue

            step_actions = []
            for tc in response.message.tool_calls:
                if (tc.function.name, tc.function.arguments) in all_actions:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": tc.function.name,
                            "content": f"Duplicate tool call",
                        }
                    )
                    continue
                step_actions.append((tc.function.name, tc.function.arguments))
                all_actions.append((tc.function.name, tc.function.arguments))

                if tc.function.name in available_functions:
                    try:
                        result = available_functions[tc.function.name](
                            **tc.function.arguments
                        )
                    except Exception as e:
                        result = f"Error: {e}"

                    # prevent context explosion
                    if len(result) > AgenticRetrieverConfig.max_tool_output_length:
                        result = f"{result[:AgenticRetrieverConfig.max_tool_output_length]}\n... TRUNCATED, TOOL OUTPUT TOO LARGE"

                    if tc.function.name == "finish":
                        answer_ids = result[:top_k]
                        finished = True

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
                            "tool_name": tc.function.name,
                            "content": f"Unknown tool",
                        }
                    )
            log_dict["steps"].append(
                {"thought": response.message.thinking, "actions": step_actions}
            )

            if finished:
                break

        log_dict["input_tokens"] = response.prompt_eval_count
        log_dict["generated_tokens"] = total_eval_count
        return answer_ids, log_dict
