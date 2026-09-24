import urllib.request

URL = "https://huggingface.co/datasets/snap-stanford/stark/raw/main/qa/amazon/stark_qa/stark_qa.csv"
OUTPUT_FILE = "../../qa_datasets/stark-amazon.csv"

urllib.request.urlretrieve(URL, OUTPUT_FILE)
