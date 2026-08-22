import csv
import json
import time
from ollama import Client

INPUT_CSV = "../graphs/stark-amazon/products.csv"
OUTPUT_CSV = "../graphs/stark-amazon/products_with_embeddings.csv"
MODEL_NAME = "qwen3-embedding:4b" 

BATCH_SIZE = 32  # Anzahl Texte pro Ollama-Anfrage

# Fix: CSV-Zellen dürfen standardmäßig 131.072 Chars groß sein
# -> Setzung auf 100.000.000 Chars
csv.field_size_limit(100000000)

def process_batch(client: Client, batch: list[dict]) -> list[dict]:
    """Nimmt Zeilen aus products.csv entgegen, holt Embeddings von Ollama und gibt erweiterte Zeilen zurück."""
    texts = [row.get("document", "") for row in batch]

    response = client.embed(model=MODEL_NAME, input=texts)
    embeddings = response["embeddings"]

    for i, row in enumerate(batch):
        row["embedding:float[]"] = ";".join(map(str, embeddings[i]))

    return batch


def main():
    client = Client()

    with open(INPUT_CSV, mode="r", encoding="utf-8") as infile, open(
        OUTPUT_CSV, mode="w", encoding="utf-8", newline=""
    ) as outfile:
        
        reader = csv.DictReader(infile)
        row_count = sum(1 for _ in csv.DictReader(infile))
        infile.seek(0)
        reader = csv.DictReader(infile)

        fieldnames = reader.fieldnames + ["embedding:float[]"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        total_processed = 0

        batch = []

        start_time = time.time()

        for row in reader:
            batch.append(row)

            # Batch voll -> Anfrage senden
            if len(batch) >= BATCH_SIZE:
                processed_batch = process_batch(client, batch)
                writer.writerows(processed_batch)
                total_processed += len(processed_batch)
                batch = []

                elapsed_time = time.time() - start_time
                eta = (elapsed_time / total_processed) * (row_count - total_processed)
                print(f"{total_processed}/{row_count}")
                print(f"Elapsed time: {elapsed_time:.1f} seconds")
                print(f"Estimated remaining time: {eta:.1f} seconds")

        # Rest-Batch verarbeiten
        if batch:
            processed_batch = process_batch(client, batch)
            writer.writerows(processed_batch)

    print("Fin.")

if __name__ == "__main__":
    main()