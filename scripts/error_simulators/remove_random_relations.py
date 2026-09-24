import yaml
import csv
import os
import random

from utils import get_deduplicated_triples_from_csv

CONFIG_FILE = "../../configs/config.yaml"
BASE_GRAPH = "../../graphs/stark-amazon"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SEED = config["error_simulation"]["seed"]
DROP_RATIO = config["error_simulation"]["remove_random_relations"]["drop_ratio"]

random.seed(SEED)
output_dir = f"{BASE_GRAPH}-rel-incomp-{SEED}"
os.makedirs(output_dir)

input_file = os.path.join(BASE_GRAPH, "triples.csv")
output_file = os.path.join(output_dir, "triples.csv")

# Deduplicate bidirectional triples
header, dedup_triples, _ = get_deduplicated_triples_from_csv(input_file)

total_triples = len(dedup_triples)
num_to_keep = int(total_triples * (1.0 - DROP_RATIO))

# Sample random triples
dedup_triples = list(dedup_triples)
print(len(dedup_triples))
keep_triples = random.sample(dedup_triples, num_to_keep)

with open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)

    # only write sampled triples
    for u, rel, v in keep_triples:
        writer.writerow([u, rel, v])
        writer.writerow([v, rel, u])
