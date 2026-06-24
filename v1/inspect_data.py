from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

def inspect_cfPers():
    with driver.session() as session:
        result = session.run("MATCH (p:cfPers) RETURN p LIMIT 1")
        for record in result:
            node = record["p"]
            print("=== cfPers properties ===")
            for key, value in node.items():
                print(f"{key}: {value}")

def inspect_cfProj():
    with driver.session() as session:
        result = session.run("MATCH (p:cfProj) RETURN p LIMIT 1")
        for record in result:
            node = record["p"]
            print("\n=== cfProj properties ===")
            for key, value in node.items():
                print(f"{key}: {value}")

inspect_cfPers()
inspect_cfProj()