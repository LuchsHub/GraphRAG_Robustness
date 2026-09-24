import yaml

from neo4j import GraphDatabase

CONFIG_FILE = "../../configs/config.yaml"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

NEO4J_URI = config["neo4j"]["uri"]
NEO4J_USER = config["neo4j"]["user"]
NEO4J_PASSWORD = config["neo4j"]["password"]
LABELS = config["graph"]["entity_labels"]
EMBED_DIMENSIONS = config["models"]["embed_dimensions"]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

for label in LABELS:
    driver.execute_query(f"""
      CREATE VECTOR INDEX {label}_index IF NOT EXISTS
      FOR (e:{label}) ON (e.embedding)
      OPTIONS {{
        indexConfig: {{
          `vector.dimensions`: {EMBED_DIMENSIONS},
          `vector.similarity_function`: 'cosine'
        }}
      }}
  """)
