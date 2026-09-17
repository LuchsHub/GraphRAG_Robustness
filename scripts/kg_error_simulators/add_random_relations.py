import csv
import os
import random
import shutil

from utils import get_deduplicated_triples_from_csv

SEED = 7
BASE_GRAPH = "../../graphs/stark-amazon"
ADDITIONAL_EDGE_RATIO = 0.50
GENERIC_RELATIONS = (
    "related to",
    "associated with",
    "connected to",
    "linked with",
)  # following CS-RAG

random.seed(SEED)

# find new output dir name and create
i = 0
while os.path.exists(f"{BASE_GRAPH}-rel-add-{i}"):
    i += 1
output_dir = f"{BASE_GRAPH}-rel-add-{i}"
os.makedirs(output_dir)

input_file = os.path.join(BASE_GRAPH, "triples.csv")
output_file = os.path.join(output_dir, "triples.csv")

# deduplicate bidirectional triples
_, dedup_triples, max_id = get_deduplicated_triples_from_csv(input_file)

total_triples = len(dedup_triples)
num_to_add = int(total_triples * ADDITIONAL_EDGE_RATIO)
print(total_triples)
print(num_to_add)

# expand base file
shutil.copyfile(input_file, output_file)

with open(output_file, "a", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)

    # add random generic edges
    added_triples = set()
    while len(added_triples) < num_to_add:
        src_id = random.randint(0, max_id)
        dst_id = random.randint(0, max_id)
        while src_id == dst_id:
            dst_id = random.randint(0, max_id)
        rel = random.choice(GENERIC_RELATIONS)

        # edges should be unique
        triple = (src_id, rel, dst_id) if src_id < dst_id else (dst_id, rel, src_id)
        if triple in added_triples:
            continue
        added_triples.add(triple)

        writer.writerow([src_id, rel, dst_id])
        writer.writerow([dst_id, rel, src_id])
