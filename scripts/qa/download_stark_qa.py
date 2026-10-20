import csv
import io
import yaml
import urllib.request

from paths import ROOT_PATH, CONFIG_PATH

OUTPUT_FILE_PATH = ROOT_PATH / "qa_datasets" / "stark-amazon.csv"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

URL = config["qa_dataset_generation"]["stark"]["url"]
NUM_QUESTIONS = config["qa_dataset_generation"]["stark"]["num_questions"]

with urllib.request.urlopen(URL) as response, open(
    OUTPUT_FILE_PATH, mode="w", newline="", encoding="utf-8"
) as out_f:

    # bytes --> text --> csv handler
    text_stream = io.TextIOWrapper(response, encoding="utf-8")
    reader = csv.reader(text_stream)
    writer = csv.writer(out_f)

    header = next(reader)
    writer.writerow(header)

    for i, row in enumerate(reader):
        if i >= NUM_QUESTIONS:
            break
        writer.writerow(row)
