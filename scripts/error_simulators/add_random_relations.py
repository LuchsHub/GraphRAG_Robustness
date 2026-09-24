import yaml
import csv
import os
import random
import shutil

from utils import get_deduplicated_triples_from_csv

CONFIG_FILE = "../../configs/config.yaml"
BASE_GRAPH = "../../graphs/stark-amazon"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SEED = config["error_simulation"]["seed"]
ADDITIONAL_EDGE_RATIO = config["error_simulation"]["add_random_relations"]["add_edge_ratio"]
RELATION_TYPES = config["error_simulation"]["add_random_relations"]["relation_types"]

random.seed(SEED)
output_dir = f"{BASE_GRAPH}-rel-add-{SEED}"
os.makedirs(output_dir)

input_file = os.path.join(BASE_GRAPH, "triples.csv")
output_file = os.path.join(output_dir, "triples.csv")

# deduplicate bidirectional triples
_, dedup_triples, max_id = get_deduplicated_triples_from_csv(input_file)

total_triples = len(dedup_triples)
num_to_add = int(total_triples * ADDITIONAL_EDGE_RATIO)

# expand base file
shutil.copyfile(input_file, output_file)

with open(output_file, "a", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)

    # add random generic edges
    added_triples = set()
    while len(added_triples) < num_to_add:
        # random nodes (can be any entity type, but mostly products since they are the majority)
        src_id = random.randint(0, max_id)
        dst_id = random.randint(0, max_id)
        # no self-loops
        while src_id == dst_id:
            dst_id = random.randint(0, max_id)
        rel = random.choice(RELATION_TYPES)

        # edges should be unique
        triple = (src_id, rel, dst_id) if src_id < dst_id else (dst_id, rel, src_id)
        if triple in added_triples:
            continue
        added_triples.add(triple)

        # add edges bidirectionally
        writer.writerow([src_id, rel, dst_id])
        writer.writerow([dst_id, rel, src_id])
