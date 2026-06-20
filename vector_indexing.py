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

EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions


def get_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def embed_persons():
    with driver.session() as session:
        result = session.run("""
            MATCH (p:cfPers) 
            WHERE p.textDescription IS NOT NULL
            RETURN p.cfPersId as id, p.cfPersNameEN as name, p.textDescription as description
        """)
        people = list(result)

    print(f"Embedding {len(people)} researchers...\n")

    for i, person in enumerate(people, 1):
        print(f"[{i}/{len(people)}] Embedding {person['name']}...")
        vector = get_embedding(person['description'])

        with driver.session() as session:
            session.run("""
                MATCH (p:cfPers {cfPersId: $id})
                SET p.embedding = $embedding
            """, id=person['id'], embedding=vector)

    print("All researchers embedded.\n")


def embed_projects():
    with driver.session() as session:
        result = session.run("""
            MATCH (p:cfProj) 
            WHERE p.textDescription IS NOT NULL
            RETURN p.cfProjId as id, p.cfTitleEN as title, p.textDescription as description
        """)
        projects = list(result)

    print(f"Embedding {len(projects)} projects...\n")

    for i, proj in enumerate(projects, 1):
        print(f"[{i}/{len(projects)}] Embedding {proj['title']}...")
        vector = get_embedding(proj['description'])

        with driver.session() as session:
            session.run("""
                MATCH (p:cfProj {cfProjId: $id})
                SET p.embedding = $embedding
            """, id=proj['id'], embedding=vector)

    print("All projects embedded.\n")


def create_vector_indexes():
    with driver.session() as session:
        # Vector index for researchers
        session.run("""
            CREATE VECTOR INDEX person_embeddings IF NOT EXISTS
            FOR (p:cfPers) ON p.embedding
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}
        """)
        # Vector index for projects
        session.run("""
            CREATE VECTOR INDEX project_embeddings IF NOT EXISTS
            FOR (p:cfProj) ON p.embedding
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}
        """)
    print("Vector indexes created.\n")


if __name__ == "__main__":
    embed_persons()
    embed_projects()
    create_vector_indexes()
    print("Vector indexing complete!")