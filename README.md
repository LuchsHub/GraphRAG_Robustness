# master_thesis
Repository for everything regarding my master's thesis.

## Pipeline 

1. Run [`notebooks/download_stark_kg.ipynb`](notebooks/download_stark_kg.ipynb) to download the STaRK-Amazon KG as `graphs/stark-amazon/nodes.csv` and `graphs/stark-amazon/triples.csv`.
2. Run [`scripts/embeddings.py`](scripts/embeddings.py) to enrich every product entity with a document-based embedding, resulting in `graphs/stark-amazon/nodes_with_embeddings.csv`.
3. Import `nodes_with_embeddings.csv` and `triples.csv` into Neo4j. I used the Neo4j Admin Import via `docker run --rm -v $HOME/neo4j/data:/data -v $HOME/neo4j/import:/import -v $HOME/neo4j/logs:/logs neo4j:latest neo4j-admin database import full neo4j --multiline-fields=true --nodes=/import/nodes_with_embeddings.csv --relationships=/import/triples.csv`.
4. Use [`notebooks/neo4j_vecindex_tools.ipynb`](notebooks/neo4j_vecindex_tools.ipynb) to create Neo4j vector indexes.
5. Create QA datasets:
	- Run [`scripts/download_stark_qa.py`](scripts/download_stark_qa.py) to download the STaRK-Amazon QA dataset.
	- Run [`scripts/generate_rel_qa_dataset.py`](scripts/generate_rel_qa_dataset.py) to generate a relational QA dataset based on the STaRK-Amazon KG.
	- Run [`notebooks/generate_text_qa_dataset.ipynb`](notebooks/generate_text_qa_dataset.ipynb) to generate a text-based QA data based on documents from STaRK-Amazon.
6. Create flawed graph variants with `scripts/kg_error_simulators`:
To be continued...

## Repository Structure

```text
├── notebooks/            # Jupyter notebooks for toolboxes and partially-repeatable scripts
├── scripts/              # Python scripts for functions and time-intensive tasks
├── graphs/               # CSV files for base graph and flawed graph versions
└── qa_datasets/          # CSV files for Question Answering datasets
