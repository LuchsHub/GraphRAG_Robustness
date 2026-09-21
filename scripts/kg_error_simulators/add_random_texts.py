import csv
import os
import random
import time

# stark_qa allows finding all product IDs and docs from given product IDs quicker than brute-force CSV iteration
from stark_qa import load_skb
from ollama import embed

DATASET_NAME = "amazon"
SEED = 7
BASE_GRAPH = "../../graphs/stark-amazon"
NUM_GRAPHS = 3
ENTITIES_COUNT = 1035542
NOISE_PROBABILITY = 0.30
MODEL_NAME = "qwen3-embedding:4b"


def add_random_chunks(
    base_doc: str,
    add_doc: str,
    rng: random.Random,
    min_chunks: int = 1,
    max_chunks: int = 3,
) -> str:
    chunks = add_doc.split("\n- ")[1:]
    num_chunks = rng.randint(min_chunks, max_chunks)
    selected_chunks = rng.sample(chunks, min(num_chunks, len(chunks)))
    return base_doc + "- " + "\n- ".join(selected_chunks)


csv.field_size_limit(100000000)

# get all product IDs
skb = load_skb(DATASET_NAME, download_processed=True)
product_ids = skb.get_node_ids_by_type("product")

input_file = os.path.join(BASE_GRAPH, "nodes_with_embeddings.csv")

# find new output dir names and create
output_files = []
dir_index = 0
for _ in range(NUM_GRAPHS):
    while os.path.exists(f"{BASE_GRAPH}-text-add-{dir_index}"):
        dir_index += 1
    output_dir = f"{BASE_GRAPH}-text-add-{dir_index}"
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

        if row[":LABEL"].startswith("product"):
            # Which output files should get a noisy row?
            noises = [rng.random() < NOISE_PROBABILITY for rng in rngs]

            # iterate through RNG results
            for j, noise in enumerate(noises):
                if noise:
                    # get two random product documents
                    p1, p2 = rngs[j].sample(product_ids, 2)
                    doc1 = skb.get_doc_info(p1, add_rel=False)
                    doc2 = skb.get_doc_info(p2, add_rel=False)

                    # prepend some chunks from doc 1
                    noisy_doc = add_random_chunks("", doc1, rngs[j])

                    # insert original doc
                    if not noisy_doc.endswith("\n"):
                        noisy_doc += "\n"
                    noisy_doc += row["document"]

                    # append some chunks from doc 2
                    noisy_doc = add_random_chunks(noisy_doc, doc2, rngs[j])

                    # re-embed node
                    response = embed(
                        model=MODEL_NAME,
                        input=[noisy_doc],
                        options={"temperature": 0.0},
                    )
                    embeddings = response["embeddings"]

                    flawed_row = row.copy()
                    if len(embeddings) > 0:
                        embedding = embeddings[0]
                        flawed_row["embedding:float[]"] = ";".join(map(str, embedding))
                        flawed_row["document"] = noisy_doc
                    writers[j].writerow(flawed_row)

                # write original row if RNG result is negative
                else:
                    writers[j].writerow(row)

        # write original row for non-products
        else:
            for writer in writers:
                writer.writerow(row)
