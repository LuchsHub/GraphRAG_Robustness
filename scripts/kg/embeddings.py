import csv
import time
import yaml

from ollama import Client

CONFIG_FILE = "../../configs/config.yaml"
INPUT_CSV = "../../graphs/stark-amazon/nodes.csv"
OUTPUT_CSV = "../../graphs/stark-amazon/nodes_with_embeddings.csv"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

EMBEDDING_MODEL = config["embedding_model"]["name"]
BATCH_SIZE = config["embedding_model"]["batch_size"]
ROW_COUNT = config["graph_entity_count"]

# Fix: CSV fields can only be 131.072 chars big
# -> set to 100.000.000 chars
csv.field_size_limit(100000000)


def process_batch(client: Client, batch: list[dict]) -> list[dict]:
    """Adds Ollama embedding to each row in a batch"""
    texts = [row.get("document", "") for row in batch]

    response = client.embed(
        model=EMBEDDING_MODEL,
        input=texts,
        options={"temperature": 0.0},
    )
    embeddings = response["embeddings"]

    for i, row in enumerate(batch):
        row["embedding:float[]"] = ";".join(map(str, embeddings[i]))

    return batch


client = Client()

with open(INPUT_CSV, mode="r", encoding="utf-8") as infile, open(
    OUTPUT_CSV, mode="w", encoding="utf-8", newline=""
) as outfile:

    reader = csv.DictReader(infile)

    # add embeddings header
    fieldnames = reader.fieldnames + ["embedding:float[]"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    total_processed = 0
    batch = []
    start_time = time.time()

    for row in reader:
        batch.append(row)

        # batch full -> send Ollama request
        if len(batch) >= BATCH_SIZE:
            processed_batch = process_batch(client, batch)
            writer.writerows(processed_batch)
            total_processed += len(processed_batch)
            batch = []

            elapsed_time = time.time() - start_time
            eta = (elapsed_time / total_processed) * (ROW_COUNT - total_processed)
            print(f"{total_processed}/{ROW_COUNT}")
            print(f"Elapsed time: {elapsed_time:.1f} seconds")
            print(f"Estimated remaining time: {eta:.1f} seconds")

    # process remaining batch
    if batch:
        processed_batch = process_batch(client, batch)
        writer.writerows(processed_batch)

print("Fin.")
