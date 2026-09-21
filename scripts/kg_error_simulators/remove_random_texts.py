import csv
import os
import random
import time
from ollama import embed

SEED = 7
BASE_GRAPH = "../../graphs/stark-amazon"
NUM_GRAPHS = 3
ENTITIES_COUNT = 1035542
DROP_RATIO = 0.30
MODEL_NAME = "qwen3-embedding:4b"

csv.field_size_limit(100000000)

input_file = os.path.join(BASE_GRAPH, "nodes_with_embeddings.csv")

# Find new output dir names and create
output_files = []
dir_index = 0
for _ in range(NUM_GRAPHS):
    while os.path.exists(f"{BASE_GRAPH}-text-incomp-{dir_index}"):
        dir_index += 1
    output_dir = f"{BASE_GRAPH}-text-incomp-{dir_index}"
    os.makedirs(output_dir)
    output_files.append(os.path.join(output_dir, "nodes_with_embeddings.csv"))
    dir_index += 1

# open NUM_GRAPHS output files
# with statement needs to be changed along the NUM_GRAPHS constant, I have not found a simpler way to do this yet
with open(input_file, "r", encoding="utf-8", newline="") as infile, open(
    output_files[0], "w", encoding="utf-8", newline=""
) as out0, open(output_files[1], "w", encoding="utf-8", newline="") as out1, open(
    output_files[2], "w", encoding="utf-8", newline=""
) as out2:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writers = [csv.DictWriter(out, fieldnames=fieldnames) for out in [out0, out1, out2]]
    for writer in writers:
        writer.writeheader()

    # initialize three seed instances
    rngs = [random.Random(SEED + i) for i in range(NUM_GRAPHS)]
    start_time = time.time()

    for i, row in enumerate(reader, 1):
        if i % 1000 == 0:
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / i) * (ENTITIES_COUNT - i)
            print(f"{i}/{ENTITIES_COUNT}")
            print(f"Estimated remaining time: {eta} seconds")

        # only remove text for products
        if row[":LABEL"].startswith("product"):
            # drop chance per seed
            drops = [rng.random() < DROP_RATIO for rng in rngs]

            # generate embedding for "name" if any seed hit
            if any(drops):
                response = embed(
                    model=MODEL_NAME,
                    input=row["name"],
                    options={"temperature": 0.0},
                )
                embeddings = response["embeddings"]

                # produce flawed row, except for error case len(embeddings) == 0
                flawed_row = row.copy()
                if len(embeddings) > 0:
                    embedding = embeddings[0]
                    flawed_row["embedding:float[]"] = ";".join(map(str, embedding))
                    flawed_row["document"] = "/"

                # write flawed row for each seed that hit, otherwise write original row
                for j, dropped in enumerate(drops):
                    writers[j].writerow(flawed_row if dropped else row)
                continue

        # write original row for non-products
        for writer in writers:
            writer.writerow(row)
