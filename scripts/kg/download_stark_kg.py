from stark_qa import load_skb
import csv

OUTPUT_DIR = "..././graphs/stark-amazon"

skb = load_skb("amazon", download_processed=True)

# for edge x, edge_index[0][x] returns ID of node 1, edge_index[1][x] returns ID of node 2
src_ids, dst_ids = skb.edge_index.tolist()

# edge_types[x] returns type ID of edge x
edge_types = skb.edge_types.tolist()

# Translating edge type IDs to corresponding strings like 1 --> "also_view"
edge_names = [skb.edge_type_dict[e] for e in edge_types]

# Create triples as (node 1 ID, edge, node 2 ID)
triples = zip(src_ids, edge_names, dst_ids)

triples_path = f"{OUTPUT_DIR}/triples.csv"
with open(triples_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([":START_ID", ":TYPE", ":END_ID"])
    writer.writerows(triples)

# Collect nodes with ID, type, name, and document
nodes = []
for n_id in range(skb.num_nodes()):
    n_info = skb.node_info[n_id]
    n_type = skb.get_node_type_by_id(n_id)
    n_name = (
        n_info.get("title")
        or n_info.get("brand_name")
        or n_info.get("color_name")
        or n_info.get("category_name")
    )
    n_doc = skb.get_doc_info(n_id, add_rel=False)

    node = (
        n_id,
        n_name,
        n_doc,
        n_type + ";entity",
    )
    nodes.append(node)

nodes_path = f"{OUTPUT_DIR}/nodes.csv"
with open(nodes_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id:ID", "name", "document", ":LABEL"])
    writer.writerows(nodes)
