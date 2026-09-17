import csv
import os
import random

from utils import get_deduplicated_triples_from_csv

BASE_GRAPH = "../../graphs/stark-amazon"
DROP_RATIO = 0.50
SEED = 7

random.seed(SEED)

# Find new output dir name and create
i = 0
while os.path.exists(f"{BASE_GRAPH}-rel-incomp-{i}"):
    i += 1
output_dir = f"{BASE_GRAPH}-rel-incomp-{i}"
os.makedirs(output_dir, exist_ok=True)

input_file = os.path.join(BASE_GRAPH, "triples.csv")
output_file = os.path.join(output_dir, "triples.csv")

# Deduplicate bidirectional triples
header, dedup_triples = get_deduplicated_triples_from_csv(input_file)

total_triples = len(dedup_triples)
num_to_keep = int(total_triples * (1.0 - DROP_RATIO))
print(total_triples)

# Sample random triples
dedup_triples = list(dedup_triples)
keep_triples = random.sample(dedup_triples, num_to_keep)
print(len(keep_triples))

# Write to output
with open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)

    for u, rel, v in keep_triples:
        writer.writerow([u, rel, v])
        writer.writerow([v, rel, u])
