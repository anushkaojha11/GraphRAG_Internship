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


def generate_person_description(name_en, keywords, research_interests_th):
    prompt = f"""
Translate the Thai research interests below into English, then write a 
single natural-language paragraph (2-3 sentences) describing this researcher.

Name: {name_en}
Keywords: {keywords}
Research Interests (Thai): {research_interests_th}

Write only the paragraph, no preamble, no labels.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def generate_project_description(title_en, keywords, abstract_th, acronym):
    prompt = f"""
Translate the Thai abstract below into English, then write a single 
natural-language paragraph (2-3 sentences) describing this research project.

Title: {title_en}
Acronym: {acronym}
Keywords: {keywords}
Abstract (Thai): {abstract_th}

Write only the paragraph, no preamble, no labels.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def index_persons():
    with driver.session() as session:
        result = session.run("""
            MATCH (p:cfPers) 
            RETURN p.cfPersId as id, p.cfPersNameEN as name, 
                   p.cfKeyw as keywords, p.cfResInt as researchInterests
        """)
        people = list(result)

    print(f"Found {len(people)} researchers to process\n")

    for i, person in enumerate(people, 1):
        print(f"[{i}/{len(people)}] Processing {person['name']}...")
        description = generate_person_description(
            person['name'], person['keywords'], person['researchInterests']
        )
        print(f"  -> {description}\n")

        with driver.session() as session:
            session.run("""
                MATCH (p:cfPers {cfPersId: $id})
                SET p.textDescription = $description
            """, id=person['id'], description=description)

    print("All researchers indexed.\n")


def index_projects():
    with driver.session() as session:
        result = session.run("""
            MATCH (p:cfProj) 
            RETURN p.cfProjId as id, p.cfTitleEN as title, 
                   p.cfKeyw as keywords, p.cfAbstr as abstract, p.cfAcro as acronym
        """)
        projects = list(result)

    print(f"Found {len(projects)} projects to process\n")

    for i, proj in enumerate(projects, 1):
        print(f"[{i}/{len(projects)}] Processing {proj['title']}...")
        description = generate_project_description(
            proj['title'], proj['keywords'], proj['abstract'], proj['acronym']
        )
        print(f"  -> {description}\n")

        with driver.session() as session:
            session.run("""
                MATCH (p:cfProj {cfProjId: $id})
                SET p.textDescription = $description
            """, id=proj['id'], description=description)

    print("All projects indexed.\n")


if __name__ == "__main__":
    index_persons()
    index_projects()
    print("Text indexing complete!")