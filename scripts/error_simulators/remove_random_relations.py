import yaml
import csv
import os
import random

from utils import get_deduplicated_triples_from_csv
from paths import ROOT_PATH, CONFIG_PATH

BASE_GRAPH = ROOT_PATH / "graphs" / "stark-amazon"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SEEDS = config["error_simulation"]["seeds"]
DROP_RATIO = config["error_simulation"]["remove_random_relations"]["drop_ratio"]

input_file = os.path.join(BASE_GRAPH, "triples.csv")

# Deduplicate bidirectional triples
header, dedup_triples, _ = get_deduplicated_triples_from_csv(input_file)

total_triples = len(dedup_triples)
num_to_keep = int(total_triples * (1.0 - DROP_RATIO))

# Sample random triples
dedup_triples = list(dedup_triples)

for seed in SEEDS:
    output_dir = f"{BASE_GRAPH}-rel-incomp-{seed}"
    try:
        os.makedirs(output_dir)
    except OSError:
        print(f"{output_dir} already exists, skipping seed.")
        continue
    output_file = os.path.join(output_dir, "triples.csv")

    # sample random triples
    random.seed(seed)
    keep_triples = random.sample(dedup_triples, num_to_keep)

    with open(output_file, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(header)

        # only write sampled triples
        for u, rel, v in keep_triples:
            writer.writerow([u, rel, v])
            writer.writerow([v, rel, u])
