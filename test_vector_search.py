from neo4j import GraphDatabase
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def search_researchers(query, top_k=5):
    query_vector = get_embedding(query)

    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('person_embeddings', $top_k, $queryVector)
            YIELD node, score
            RETURN node.cfPersNameEN as name, node.textDescription as description, score
        """, top_k=top_k, queryVector=query_vector)

        print(f"\nQuery: '{query}'")
        print("=" * 60)
        for record in result:
            print(f"\n{record['name']} (score: {record['score']:.4f})")
            print(f"  {record['description']}")


search_researchers("Who knows about machine vision for fruit grading?")